"""Check grouping, feature alignment and bootstrap with synthetic inputs."""
import csv
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import load,dirichlet_clients
from bootstrap import grouped_bootstrap


class ReconstructionTests(unittest.TestCase):
    def test_perfect_predictions_and_group_preservation(self):
        y=np.repeat(np.arange(4),4);groups=np.array([f'g{i//2}' for i in range(16)])
        p=np.eye(4)[y]
        summary,draws=grouped_bootstrap(y,{'perfect':p},groups,n_rep=20)
        for ci in summary['ci95']['perfect'].values():self.assertEqual(ci,[1.0,1.0])
        self.assertEqual(len(draws),20)
        clients=dirichlet_clients(y,groups)
        for g in np.unique(groups):self.assertEqual(len(set(clients[groups==g])),1)

    def test_conflicting_group_labels_rejected(self):
        y=np.arange(4);p=np.eye(4)
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            grouped_bootstrap(y,{'model':p},np.array(['same','same','b','c']),n_rep=2)

    def test_feature_row_alignment_and_split_leakage_rejected(self):
        names=['COVID','Lung_Opacity','Normal','Viral Pneumonia']
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);manifest=path/'manifest.csv';features=path/'features.npz'
            rows=[{'path':f'{i}.png','class':names[i%4],'group_id':f'g{i}','split':['train','val','test'][i//4]} for i in range(12)]
            def write():
                with manifest.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            write();np.savez(features,features=np.ones((12,512)),paths=np.array([r['path'] for r in rows]))
            self.assertEqual(load(manifest,features)[0].shape,(12,512))
            np.savez(features,features=np.ones((12,512)),paths=np.array([r['path'] for r in rows][::-1]))
            with self.assertRaisesRegex(ValueError,'row order'):load(manifest,features)
            np.savez(features,features=np.ones((12,512)));rows[4]['group_id']='g0';write()
            with self.assertRaisesRegex(ValueError,'crosses splits'):load(manifest,features)


if __name__=='__main__':unittest.main()
