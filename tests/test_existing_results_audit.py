"""Offline audit contracts: duplicate evidence, pairing and event accounting."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lfba_comparison.results_audit import deduplicate, aggregate, paired, scan, snapshot, verify_snapshot, normalize, update_batch_counts
from lfba_comparison.audit_diagnostics import exposure_statistics, diagnose


def record(**extra):
    row = dict(source='a', batch='batch', dataset='CIFAR10', seed=1, method='auto_vote',
               status='complete', category='formal', protocol_group='p', poison_rate=.1,
               select_rate=.5, training_epochs=40, evaluation_epoch=40, endpoint='last',
               data_digest='data', candidate_digest='choice', final_model_digest='model',
               start_state='start', target_class=0, anchor=1, acc=.8, asr=.9)
    row.update(extra)
    return row


class AuditTests(unittest.TestCase):
    def test_equal_metrics_without_model_evidence_do_not_prove_duplicate(self):
        rows = [record(final_model_digest=None), record(source='b', final_model_digest=None)]
        deduplicate(rows)
        self.assertTrue(all(r['duplicate_of'] is None for r in rows))
        self.assertEqual(aggregate(rows)[0]['n'], 0)
        self.assertEqual(aggregate(rows)[0]['ambiguous_seeds'], [1])

    def test_reused_model_is_counted_once_even_with_different_prefix_digests(self):
        rows = [record(), record(source='b', start_state='three-round-prefix')]
        deduplicate(rows)
        self.assertEqual(rows[1]['duplicate_of'], 'a')
        self.assertEqual(aggregate(rows)[0]['n'], 1)
        self.assertIsNone(aggregate(rows)[0]['asr_std'])

    def test_missing_scores_are_not_zero_and_pairs_require_shared_state(self):
        rows = [record(method='lfba_original', asr=.4,final_model_digest='baseline'), record(source='b'),
                record(source='c', seed=2, asr=None),
                record(source='d', seed=2, method='lfba_original', start_state='other')]
        deduplicate(rows)
        pairs = paired(rows)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]['delta_asr_pp'], 50.)
        self.assertEqual(next(g for g in aggregate(rows) if g['method']=='auto_vote')['n'], 1)

    def test_repeated_recipient_events_and_donor_labels_are_counted(self):
        result = exposure_statistics([0, 0, 1, 1], [0, 2], 0,
            [dict(epoch=4, recipients=[0, 2], donors=[1, 3]),
             dict(epoch=5, recipients=[2], donors=[1])])
        self.assertEqual(result['wrong_recipient_events'], 2)
        self.assertEqual(result['replacement_events'], 3)
        self.assertEqual(result['unique_recipients'], 2)
        self.assertEqual(result['max_recipient_uses'], 2)
        self.assertEqual(result['donor_class_events'], {'0': 2, '1': 1})
        self.assertEqual(result['candidate_purity_recomputed'], .5)

    def test_scan_distinguishes_missing_corrupt_and_low_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); batch = root/'results'/'batch'; batch.mkdir(parents=True)
            (batch/'protocol.json').write_text(json.dumps(dict(dataset='CIFAR10', seeds=[1],
                methods=['auto_vote','lfba_original','auto_mean'], epochs=40, smoke=False)))
            arm=batch/'seed_1'/'auto_vote'; arm.mkdir(parents=True)
            (arm/'metrics.json').write_text('{broken')
            rows, batches, issues = scan(root, root/'output')
            self.assertEqual(sorted(r['status'] for r in rows), ['corrupt','missing','missing'])
            self.assertEqual(batches[0]['complete'], 0)
            self.assertTrue(issues)

    def test_input_hash_verification_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); f=root/'metrics.json'; f.write_text('{}')
            before=snapshot([f]); verify_snapshot(before)
            f.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_snapshot(before)

    def test_standalone_inventory_keeps_the_same_batch_identity_when_recounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm=root/'results'/'CIFAR10'/'timestamp';arm.mkdir(parents=True)
            (arm/'metrics.json').write_text(json.dumps(dict(test={'acc':.8,'asr':.2},checkpoint_epoch=2)))
            (arm/'config.json').write_text(json.dumps(dict(dataset='CIFAR10',seed=1,attack='LFBA')))
            rows,batches,_=scan(root,root/'output')
            deduplicate(rows);update_batch_counts(rows,batches)
            self.assertEqual(batches[0]['complete'],1)
            self.assertEqual(rows[0]['category'],'standalone_training')

    def test_missing_entire_matrix_case_is_registered_without_writing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); batch=root/'results'/'matrix';batch.mkdir(parents=True)
            (batch/'protocol.json').write_text(json.dumps(dict(kind='stability',dataset='CIFAR10',
                methods=['auto_vote','lfba_original'],seeds=[1,2],epochs=40,
                cases=[dict(id='p_small',poison_rate=.01,select_rate=.5)])))
            rows,batches,_=scan(root,root/'output')
            self.assertEqual(len(rows),4)
            self.assertEqual(next(b for b in batches if b.get('rollup'))['missing'],4)
            self.assertFalse((batch/'p_small').exists())

    def test_conflicting_same_seed_records_cannot_enter_paired_comparison(self):
        rows=[record(method='lfba_original'),record(source='b'),record(source='c',final_model_digest='other')]
        deduplicate(rows)
        self.assertEqual(paired(rows),[])

    def test_same_anchor_protocol_does_not_pair_different_targets(self):
        rows=[record(method='lfba_original',anchor_source='auto_vote'),
              record(source='b',target_class=1)]
        deduplicate(rows)
        self.assertEqual(paired(rows),[])

    def test_budget_and_independent_epoch_corruption_are_not_performance_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=dict(test={'acc':.8,'asr':0},epoch=4,method='auto_vote',
                cumulative_replacements=3,independent_evaluation=dict(epoch=3,test={'acc':.8,'asr':0}))
            config=dict(dataset='CIFAR10',seed=1,anchor_idx=0,target_label=0,
                        data_state={'train_ids':[0,1]})
            data=dict(history=[dict(epoch=i+1) for i in range(4)],choice={'anchor_id':0,'candidates':[0]},
                plans=[dict(epoch=4,recipients=[0],donors=[1])])
            r=normalize(root,root,{'epochs':4},'formal',root,result,config,data)
            self.assertEqual(r['status'],'corrupt')
            result['cumulative_replacements']=1
            result['independent_evaluation']['epoch']=4
            r=normalize(root,root,{'epochs':4},'formal',root,result,config,data)
            self.assertEqual(r['status'],'complete')
            self.assertEqual(r['asr'],0)

    def test_conflicting_metrics_for_same_model_are_excluded(self):
        rows=[record(),record(source='b',asr=.1)]
        deduplicate(rows)
        self.assertTrue(all(r['status']=='corrupt' for r in rows))
        self.assertEqual(aggregate(rows),[])
        batches=[dict(batch='batch',expected=2)]
        update_batch_counts(rows,batches)
        self.assertEqual((batches[0]['complete'],batches[0]['corrupt']),(0,2))

    def test_formal_record_missing_evidence_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            row=normalize(root,root,{},'formal',root,
                dict(test={'acc':.8,'asr':.9},epoch=40,method='auto_vote'),{},dict(history=[{}]*40))
            self.assertEqual(row['status'],'corrupt')

    def test_invalid_har_fingerprint_does_not_leave_partial_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data';train=data/'UCI-HAR/UCI HAR Dataset/train'
            train.mkdir(parents=True)
            np.savetxt(train/'y_train.txt',[1,2],fmt='%d')
            np.savetxt(train/'X_train.txt',np.zeros((2,561)))
            arm=root/'arm';arm.mkdir()
            (arm/'config.json').write_text(json.dumps(dict(trigger_dimensions=[281],
                data_state=dict(train_ids=[0,1],fingerprints={'source':'wrong'}))))
            (arm/'experiment_data.json').write_text(json.dumps(dict(choice={'candidates':[0]},
                plans=[dict(epoch=4,recipients=[0],donors=[1])],
                history=[dict(epoch=4,validation={'asr':.2})])))
            row=record(source=str(arm),batch='results/ucihar_lfba_vs_vote_p01_s05_v1',
                dataset='UCIHAR',anchor=0,target_class=0,cumulative_replacements=1)
            issues=[]
            diagnoses,curves,triggers,_=diagnose([row],data,issues)
            self.assertEqual((diagnoses,curves,triggers),([],[],[]))
            self.assertIn('fingerprint',issues[0]['error'])

    def test_failed_refresh_invalidates_its_previous_success_receipt(self):
        from scripts.audit_existing_results import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'results'/'audit';out.mkdir(parents=True)
            (out/'audit_config.json').write_text(json.dumps(dict(root=str(root),training=False,inference=False)))
            (out/'verification.json').write_text('{"original_inputs_unchanged":true}')
            with patch('scripts.audit_existing_results.plot',side_effect=RuntimeError('plot failed')):
                with self.assertRaisesRegex(RuntimeError,'plot failed'):
                    main(['--root',str(root),'--output',str(out),'--refresh-audit'])
            receipt=json.loads((out/'verification.json').read_text())
            self.assertEqual(receipt['status'],'in_progress')
            self.assertIsNone(receipt['original_inputs_unchanged'])


if __name__ == '__main__':
    unittest.main()
