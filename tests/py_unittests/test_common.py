import unittest
import sys
import types

# Provide a dummy z3 module so kmax.common can be imported without the real
# dependency being installed.
sys.modules.setdefault('z3', types.SimpleNamespace())
from kmax.common import FileChangeType

class DummyHeader:
    def __init__(self, old_path, new_path):
        self.old_path = old_path
        self.new_path = new_path

class DummyDiff:
    def __init__(self, old_path, new_path, text='', changes=False):
        self.header = DummyHeader(old_path, new_path)
        self.text = text
        self.changes = changes

class TestFileChangeType(unittest.TestCase):
    def test_permission_changed_new_file_mode(self):
        diff = DummyDiff('a', 'a', text='new file mode 100644')
        self.assertEqual(FileChangeType.getType(diff), FileChangeType.PERMISSION_CHANGED)

    def test_permission_changed_new_mode(self):
        diff = DummyDiff('a', 'a', text='new mode 100644')
        self.assertEqual(FileChangeType.getType(diff), FileChangeType.PERMISSION_CHANGED)

if __name__ == '__main__':
    unittest.main()
