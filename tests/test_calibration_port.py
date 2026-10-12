import socket

import pytest

from pd_disaggregation_demo.calibration import validate_engine_port


def test_completed_connection_does_not_block_next_owned_engine():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        with pytest.raises(OSError):
            validate_engine_port(port)
        with socket.create_connection(("127.0.0.1", port)) as client:
            accepted, _ = listener.accept()
            with accepted:
                accepted.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
    # A fresh server can reuse its address after the accepted connection closes.
    validate_engine_port(port)
