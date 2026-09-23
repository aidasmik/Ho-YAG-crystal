import threading
import time
from hoyag.live_control import request_control,ControlState,CancellationFlag,UserCancelled


def test_pause_step_resume_and_cancel_at_iteration_boundary(tmp_path):
    control=ControlState(tmp_path)
    request_control(tmp_path,'pause')
    done=threading.Event()
    thread=threading.Thread(target=lambda:(control.after_iteration(),done.set()))
    thread.start()
    time.sleep(.12)
    assert not done.is_set()
    request_control(tmp_path,'step')
    thread.join(timeout=1)
    assert done.is_set() and control.paused
    request_control(tmp_path,'resume')
    control.after_iteration()
    assert not control.paused
    request_control(tmp_path,'cancel')
    assert CancellationFlag(tmp_path).is_set()
    try:
        control.after_iteration()
    except UserCancelled:
        pass
    else:
        raise AssertionError('cancel was not delivered')
