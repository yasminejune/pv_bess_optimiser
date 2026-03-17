"""Tests for terminal_output module."""

from src.ors.utils.terminal_output import print_error, print_info, print_success, print_warning


def test_print_info_verbose(capsys):
    print_info("test message")
    assert "Info: test message" in capsys.readouterr().out


def test_print_info_silent(capsys):
    print_info("test message", verbose=False)
    assert capsys.readouterr().out == ""


def test_print_success(capsys):
    print_success("done")
    assert "Success: done" in capsys.readouterr().out


def test_print_warning(capsys):
    print_warning("caution")
    assert "Warning: caution" in capsys.readouterr().out


def test_print_error(capsys):
    print_error("oops")
    assert "Error: oops" in capsys.readouterr().out
