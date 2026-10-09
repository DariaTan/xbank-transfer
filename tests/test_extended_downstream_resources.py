import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / 'environments/extended_downstream_resources.sh'


class ResourcePolicyTests(unittest.TestCase):
    def shell(self, body):
        return subprocess.run(['bash', '-c', 'set -euo pipefail; source "$1"; ' + body,
                               'resource-test', str(HELPER)], text=True, capture_output=True)

    def test_only_mbd_chronos_gets_larger_budget(self):
        result = self.shell('''
task_memory_gib training.tune_mlp --evaluation mbd_raw --model chronos2
task_memory_gib training.tune_mlp --model chronos2 --evaluation mbd_daily
task_memory_gib training.tune_mlp --evaluation xbank_pair --model chronos2
task_memory_gib training.tune_mlp --evaluation mbd_raw --model mlm
task_memory_gib training.chronos_fgw lightgbm --device gpu
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ['40', '40', '24', '24', '24'])

    def test_admission_uses_actual_other_cap(self):
        result = self.shell('''
required_available_kib 40 0
required_available_kib 40 $((24*1024*1024*1024))
required_available_kib 40 $((40*1024*1024*1024))
required_available_kib 24 $((40*1024*1024*1024))
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([int(x)//1024//1024 for x in result.stdout.split()], [48, 72, 88, 72])

    def test_reservations_ignore_foreign_containers(self):
        result = self.shell('''
docker() {
 if [[ "$1" == ps ]]; then printf '%s\n' extended-downstream-gpu0 foreign-job extended-downstream-gpu1;
 elif [[ "$4" == extended-downstream-gpu0 ]]; then echo $((40*1024*1024*1024));
 elif [[ "$4" == extended-downstream-gpu1 ]]; then echo $((24*1024*1024*1024));
 else return 9; fi
}
reserved_worker_bytes
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(result.stdout.strip()), 64*1024**3)

    def test_unbounded_own_worker_is_rejected(self):
        result = self.shell('''
docker() { if [[ "$1" == ps ]]; then echo extended-downstream-gpu0; else echo 0; fi; }
reserved_worker_bytes
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no known RAM bound', result.stderr)

    def update_case(self, name='/extended-downstream-gpu0', evaluation='mbd_raw', model='chronos2',
                    ram=92, limit=24, running='true'):
        return self.shell(f'''
docker() {{
 if [[ "$1" == update ]]; then echo "MUTATION $*" >&2; return 0; fi
 case "$3" in
  *State.Running*) echo '{name} {running} {limit*1024**3} 1234' ;;
  *Config.Cmd*) printf '%s\\n' python -u -m training.tune_mlp --evaluation {evaluation} --model {model} ;;
  *) return 9 ;;
 esac
}}
available_ram_kib() {{ echo {ram*1024**2}; }}
reserved_worker_bytes() {{ echo {limit*1024**3 + 24*1024**3}; }}
date() {{ echo NOW; }}
upgrade_chronos_container immutable-id
''')

    def test_live_update_targets_immutable_id_without_restart(self):
        result = self.update_case()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('UPGRADED', result.stdout)
        self.assertIn('PID=1234', result.stdout)
        self.assertIn('MUTATION update --memory=40g --memory-swap=40g immutable-id', result.stderr)
        self.assertNotIn(' restart ', result.stderr)

    def test_live_update_waits_for_ram_without_mutation(self):
        result = self.update_case(ram=60)
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn('WAIT RAM', result.stdout)
        self.assertNotIn('MUTATION', result.stderr)

    def test_live_update_is_idempotent_and_scoped(self):
        for kwargs in [dict(limit=40), dict(limit=48), dict(name='/foreign-job'),
                       dict(model='mlm'), dict(evaluation='xbank_pair'), dict(running='false')]:
            with self.subTest(**kwargs):
                result = self.update_case(**kwargs)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn('MUTATION', result.stderr)


if __name__ == '__main__':
    unittest.main()
