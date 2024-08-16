configure_file(test_pyclient.in test_pyclient.sh @ONLY)
add_test(pyclient test_pyclient.sh)
