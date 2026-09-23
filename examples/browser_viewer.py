"""Localhost trame browser view of accepted solver snapshots or replay state."""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def serve(source: Path, *, live=False, port=8080):
    import pyvista as pv
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageLayout
    from trame.widgets import vtk, vuetify3 as v3
    from hoyag.snapshots import newest_snapshot,snapshot_from_case
    from hoyag.replay_viewer import _populate
    from hoyag.live_control import request_control

    source=source.resolve()
    snapshot=(newest_snapshot(source/'snapshots') if live else snapshot_from_case(source))
    if snapshot is None:
        raise ValueError('no accepted live snapshot is available yet')
    plotter=pv.Plotter(shape=(3,3),off_screen=True,window_size=(960,720))
    current_view='overview'
    _populate(plotter,pv,snapshot,0,0,1.,current_view)
    server=get_server('hoyag-scientific-viewer',client_type='vue3')
    server.state.status=f"{snapshot.metadata['fidelity_mode']} | {snapshot.metadata['time_kind']} | {snapshot.metadata['state_id']}"
    current_id=snapshot.metadata['state_id']

    def refresh():
        nonlocal current_id,snapshot
        fresh=newest_snapshot(source/'snapshots') if live else snapshot_from_case(source)
        if fresh is None:return
        if fresh.metadata['state_id']!=current_id:
            snapshot=fresh;current_id=fresh.metadata['state_id']
            plotter.clear()
            _populate(plotter,pv,fresh,0,0,1.,current_view)
            view.update()
        age=max(0,time.time()-snapshot.metadata['t_published_wall'])
        server.state.status=(f"{snapshot.metadata['fidelity_mode']} | "
            f"{snapshot.metadata['time_kind']} | {current_id} | state age {age:.1f} s | "
            'simulated seconds unavailable')

    def control(command):
        request_control(source,command)
        server.state.control=f'{command} requested; applies at next solver boundary'

    def select_view(name):
        nonlocal current_view
        current_view=name
        plotter.clear()
        _populate(plotter,pv,snapshot,0,0,1.,current_view)
        view.update()

    with SinglePageLayout(server) as layout:
        layout.title.set_text('Ho:YAG numerical state')
        with layout.toolbar:
            v3.VSpacer()
            v3.VBtn('Overview',click=lambda:select_view('overview'))
            v3.VBtn('Thermal',click=lambda:select_view('thermal'))
            v3.VBtn('Optics',click=lambda:select_view('optics'))
            v3.VBtn('Refresh',click=refresh)
            if live:
                v3.VBtn('Pause',click=lambda:control('pause'))
                v3.VBtn('Step',click=lambda:control('step'))
                v3.VBtn('Resume',click=lambda:control('resume'))
                v3.VBtn('Cancel',click=lambda:control('cancel'))
        with layout.content:
            with v3.VContainer(fluid=True,classes='pa-0 fill-height'):
                v3.VAlert(text=('status',),type='info',density='compact')
                if live:
                    v3.VAlert(text=('control', 'Controls apply at outer-iteration boundaries'),
                              type='warning',density='compact')
                view=vtk.VtkRemoteView(plotter.ren_win,interactive_ratio=1)

    if live:
        async def poll(**_):
            while True:
                await asyncio.sleep(.5)
                refresh()
        server.controller.on_server_ready.add_task(poll)
    server.controller.on_server_ready.add(view.update)
    server.start(host='127.0.0.1',port=port,open_browser=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case_directory',type=Path)
    p.add_argument('--live',action='store_true')
    p.add_argument('--port',type=int,default=8080)
    args=p.parse_args()
    serve(args.case_directory,live=args.live,port=args.port)


if __name__=='__main__':main()
