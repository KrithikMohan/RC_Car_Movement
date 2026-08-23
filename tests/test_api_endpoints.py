import json
import pytest
from pi_server import app, camera_grabber, drive_mode, stop_auto_pipeline


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
    stop_auto_pipeline()


def test_telemetry_endpoint(client):
    res = client.get('/telemetry')
    assert res.status_code == 200
    data = res.get_json()
    assert 'drive_mode' in data
    assert 'battery_pct' in data
    assert 'rccar_available' in data
    assert data['rccar_available'] is True


def test_speed_endpoint(client):
    res = client.post('/speed', json={'mode': 'high'})
    assert res.status_code == 200
    data = res.get_json()
    assert data['speed_mode'] == 'high'

    res = client.post('/speed', json={'mode': 'low'})
    assert res.status_code == 200
    data = res.get_json()
    assert data['speed_mode'] == 'low'


def test_mode_and_manual_override(client, monkeypatch):
    # Auto mode requires a live camera feed (see start_auto_pipeline's
    # has_camera() guard); this test is about mode-switching/override
    # behavior, not hardware availability, so fake a live camera.
    monkeypatch.setattr(camera_grabber, "has_camera", lambda: True)

    # Enable Auto Mode
    res = client.post('/mode', json={'mode': 'auto'})
    assert res.status_code == 200
    assert res.get_json()['mode'] == 'auto'

    # Verify manual control command immediately preempts auto mode
    res = client.post('/control', json={'command': 'b'})
    assert res.status_code == 200
    assert res.get_json()['drive_mode'] == 'manual'
