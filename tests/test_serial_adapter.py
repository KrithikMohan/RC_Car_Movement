import json
import pytest
from pi_server import UGV02SerialAdapter, send_serial


def test_serial_adapter_speed_mapping(monkeypatch):
    sent_commands = []
    def mock_send_serial(cmd):
        sent_commands.append(cmd)

    monkeypatch.setattr("pi_server.send_serial", mock_send_serial)

    adapter = UGV02SerialAdapter()

    # 1. Full speed forward (speed_tier=2, steer=0)
    adapter.write(b"S,2,0\n")
    assert len(sent_commands) == 1
    data = json.loads(sent_commands[-1])
    assert data["T"] == 13
    assert pytest.approx(data["X"], 0.001) == -0.350
    assert pytest.approx(data["Z"], 0.001) == 0.000

    # 2. Slow speed forward (speed_tier=1, steer=0)
    adapter.write(b"S,1,0\n")
    data = json.loads(sent_commands[-1])
    assert pytest.approx(data["X"], 0.001) == -0.200
    assert pytest.approx(data["Z"], 0.001) == 0.000

    # 3. Stop (speed_tier=0, steer=0)
    adapter.write(b"S,0,0\n")
    data = json.loads(sent_commands[-1])
    assert pytest.approx(data["X"], 0.001) == 0.000
    assert pytest.approx(data["Z"], 0.001) == 0.000


def test_serial_adapter_steer_kinematics(monkeypatch):
    sent_commands = []
    def mock_send_serial(cmd):
        sent_commands.append(cmd)

    monkeypatch.setattr("pi_server.send_serial", mock_send_serial)

    adapter = UGV02SerialAdapter()

    # Full speed, Steer left 50% (-50) -> should map to positive Z in differential kinematics
    adapter.write(b"S,2,-50\n")
    data = json.loads(sent_commands[-1])
    assert pytest.approx(data["X"], 0.001) == -0.350
    assert data["Z"] > 0.0
    assert pytest.approx(data["Z"], 0.001) == 0.700

    # Full speed, Steer right 50% (+50) -> should map to negative Z
    adapter.write(b"S,2,50\n")
    data = json.loads(sent_commands[-1])
    assert pytest.approx(data["X"], 0.001) == -0.350
    assert data["Z"] < 0.0
    assert pytest.approx(data["Z"], 0.001) == -0.700


def test_serial_adapter_version_header_ignored(monkeypatch):
    sent_commands = []
    def mock_send_serial(cmd):
        sent_commands.append(cmd)

    monkeypatch.setattr("pi_server.send_serial", mock_send_serial)

    adapter = UGV02SerialAdapter()
    adapter.write(b"V,1\n")
    assert len(sent_commands) == 0
