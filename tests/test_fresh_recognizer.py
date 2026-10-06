"""The ``fresh_recognizer`` fixture leaves no ``recognizer`` module behind.

The two tests run in definition order. The first imports ``recognizer`` through
the fixture; the second checks that the import did not outlive the first test,
where a plain ``import recognizer`` would pick up that test's environment.
"""

import sys


def test_imports_recognizer_with_this_tests_environment(fresh_recognizer):
    recognizer = fresh_recognizer(WXDU_SHAZAM_SECRET="first-test-secret")

    assert recognizer.API_SECRET == "first-test-secret"
    assert sys.modules["recognizer"] is recognizer


def test_teardown_removed_the_module_the_previous_test_imported():
    assert "recognizer" not in sys.modules
