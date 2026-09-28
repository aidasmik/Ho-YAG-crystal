"""Bounded, simulation-oracle SLM response calibration and verified correction.

The complex output fields used here are privileged solver truth. Hardware use
requires a calibrated complex-field measurement or phase retrieval stage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .field_metrics import field_metrics, phase_support


@dataclass
class TeacherResult:
    step_rad: np.ndarray | None
    field: np.ndarray | None
    phase_rms_rad: float | None
    shape_overlap: float | None
    checks: list[dict]
    evaluations: int
    selected_candidate: str | None = None


def response_modes(x, y, waist_m, *, adaptive=None, extra_modes=(), max_modes=10):
    """Low-order and local SLM directions, normalized in the illuminated pupil."""
    x=np.asarray(x,float)/float(waist_m)
    y=np.asarray(y,float)/float(waist_m)
    if x.shape!=y.shape or x.ndim!=2 or waist_m<=0 or max_modes<1:
        raise ValueError("aligned 2-D coordinates and positive waist required")
    r2=x*x+y*y
    pupil=np.exp(-2*r2)
    support=pupil>=1e-3
    raw=[("tilt_x",x),("tilt_y",y),("defocus",r2),
         ("astig_0",x*x-y*y),("astig_45",2*x*y)]
    if adaptive is not None:
        adaptive=np.asarray(adaptive,float)
        if adaptive.shape!=x.shape or not np.all(np.isfinite(adaptive)):
            raise ValueError("adaptive response direction must be finite and aligned")
        raw.insert(0,("adaptive_backprop",adaptive))
    for name,extra in reversed(tuple(extra_modes)):
        extra=np.asarray(extra,float)
        if extra.shape!=x.shape or not np.all(np.isfinite(extra)):
            raise ValueError("extra response directions must be finite and aligned")
        raw.insert(0,(name,extra))
    for sx in (-0.65,0.65):
        for sy in (-0.65,0.65):
            raw.append((f"local_{sx:+.2f}_{sy:+.2f}",
                        np.exp(-((x-sx)**2+(y-sy)**2)/0.45)))
    raw.extend((("coma_x",x*r2),("coma_y",y*r2)))
    modes=[]
    for name,raw_mode in raw:
        mode=np.where(support,raw_mode,0.)
        mean=np.sum(pupil*mode)/np.sum(pupil)
        mode=np.where(support,mode-mean,0.)
        rms=np.sqrt(np.sum(pupil*mode**2)/np.sum(pupil))
        if rms>1e-8:
            modes.append((name,mode/rms))
        if len(modes)>=max_modes:
            break
    return modes


def _metrics(field, desired, weights, phase_valid, desired_intensity):
    measured=field_metrics(field,desired,weights,phase_valid)
    return measured.phase_rms_rad,measured.shape_overlap


def response_matrix_teacher(baseline, desired, command, modes, simulate, *,
                            max_evaluations=16, probe_rad=0.06,
                            max_step_rad=0.45, ridge_fraction=0.03,
                            amplitude_weight=0.15,
                            min_phase_improvement_rad=0.05,
                            min_shape_overlap=0.95,
                            max_shape_drop=0.012,
                            trial_gains=(1.,0.5,0.25),
                            fallback_steps=(), adaptive_line_search=False,
                            pairwise_preview=False):
    """Probe the *full* forward model, fit a local response, and verify trials.

    ``simulate(command)`` returns ``(complex_output_field, physically_valid)``.
    All probes and trial evaluations share one budget. The best passing trial
    is returned; no linearized prediction is accepted as a label.
    """
    baseline=np.asarray(baseline,complex)
    desired=np.asarray(desired,complex)
    command=np.asarray(command,float)
    if baseline.shape!=desired.shape or baseline.shape!=command.shape:
        raise ValueError("baseline, desired and command must have aligned shapes")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(desired)):
        raise ValueError("complex fields must be finite")
    if max_evaluations<2 or probe_rad<=0 or max_step_rad<=0:
        raise ValueError("response teacher needs a positive probe and at least two evaluations")
    if not (0<=amplitude_weight and ridge_fraction>=0):
        raise ValueError("response fit penalties must be nonnegative")
    if not trial_gains or any(not 0<g<=1 for g in trial_gains):
        raise ValueError("trial gains must be in (0, 1]")
    fallback_steps=tuple(fallback_steps)
    for name,fallback in fallback_steps:
        fallback=np.asarray(fallback,float)
        if fallback.shape!=command.shape or not np.all(np.isfinite(fallback)):
            raise ValueError("fallback steps must be finite and aligned")
    weights=abs(baseline)**2
    phase_valid=phase_support(baseline,desired)
    if weights.max()<=0 or not np.any(phase_valid) or not np.any(desired):
        raise ValueError("baseline has no illuminated phase pixels")
    desired_intensity=abs(desired)**2
    before_rms,before_shape=_metrics(baseline,desired,weights,phase_valid,
                                     desired_intensity)
    shape_requirement=max(min_shape_overlap,before_shape-max_shape_drop)
    sample_weight=weights[phase_valid]
    sample_weight=sample_weight/max(np.sum(sample_weight),1e-30)
    residual=np.angle(baseline*np.conj(desired))
    piston=np.angle(np.sum(weights[phase_valid]*np.exp(1j*residual[phase_valid])))
    phase_error=np.angle(np.exp(1j*(residual[phase_valid]-piston)))
    floor=max(float(np.max(abs(baseline)))*1e-6,1e-30)
    amp_error=np.log(np.maximum(abs(baseline[phase_valid]),floor)/
                     np.maximum(abs(desired[phase_valid]),floor))
    amp_error-=np.sum(sample_weight*amp_error)
    checks=[]
    phase_columns=[]; amplitude_columns=[]; complex_columns=[]; used_modes=[]
    reserved_trials=min(max_evaluations-1,len(trial_gains)+len(fallback_steps))
    evaluation_limit=min(max_evaluations-reserved_trials,len(modes))
    for name,mode in modes[:evaluation_limit]:
        mode=np.asarray(mode,float)
        if mode.shape!=command.shape or not np.all(np.isfinite(mode)):
            raise ValueError("response modes must be finite and aligned")
        probe_command=np.mod(command+probe_rad*mode,2*np.pi)
        try:
            probe,valid=simulate(probe_command)
            probe=np.asarray(probe,complex)
            if probe.shape!=baseline.shape or not np.all(np.isfinite(probe)) or not valid:
                raise ValueError("invalid physical probe response")
            phase_change=np.angle(probe[phase_valid]*np.conj(baseline[phase_valid]))/probe_rad
            phase_change-=np.sum(sample_weight*phase_change)
            amplitude_change=np.log(np.maximum(abs(probe[phase_valid]),floor)/
                                    np.maximum(abs(baseline[phase_valid]),floor))/probe_rad
            amplitude_change-=np.sum(sample_weight*amplitude_change)
            if np.sqrt(np.sum(sample_weight*phase_change**2))<1e-5:
                checks.append({"kind":"probe","mode":name,"passed":False,
                               "reason":"unobservable phase response"})
                continue
            phase_columns.append(phase_change)
            amplitude_columns.append(amplitude_change)
            complex_columns.append((probe-baseline)/probe_rad)
            used_modes.append(mode)
            checks.append({"kind":"probe","mode":name,"passed":True,
                           "phase_response_rms_rad_per_rad":float(np.sqrt(
                               np.sum(sample_weight*phase_change**2)))})
        except Exception as exc:
            checks.append({"kind":"probe","mode":name,"passed":False,
                           "error":f"{type(exc).__name__}: {exc}"})
    evaluations=len(checks)
    best=None
    trial_steps=[]
    if used_modes:
        try:
            phase_matrix=np.stack(phase_columns,axis=1)
            amplitude_matrix=np.stack(amplitude_columns,axis=1)
            gram=phase_matrix.T@(sample_weight[:,None]*phase_matrix)
            gram+=amplitude_weight*(amplitude_matrix.T@(
                sample_weight[:,None]*amplitude_matrix))
            diag_scale=max(float(np.trace(gram))/len(used_modes),1e-12)
            gram+=np.eye(len(used_modes))*ridge_fraction*diag_scale
            rhs=-(phase_matrix.T@(sample_weight*phase_error)+
                  amplitude_weight*amplitude_matrix.T@(sample_weight*amp_error))
            coefficients=np.linalg.solve(gram,rhs)
            step=np.einsum("i,ijk->jk",coefficients,np.stack(used_modes),optimize=True)
            peak=float(np.max(abs(step)))
            scale=min(1.,max_step_rad/max(peak,1e-30))
            step*=scale
            fit_phase_prediction=phase_error+phase_matrix@(coefficients*scale)
            predicted_direction=np.einsum("i,ijk->jk",coefficients*scale,
                                          np.stack(complex_columns),optimize=True)
            chosen_gains=list(trial_gains)
            if adaptive_line_search:
                preview=[]
                for gain in np.linspace(.2,1.,17):
                    predicted_rms,predicted_overlap=_metrics(
                        baseline+gain*predicted_direction,desired,weights,
                        phase_valid,desired_intensity)
                    preview.append((float(gain),predicted_rms,predicted_overlap))
                preview.sort(key=lambda item:(item[2]>=shape_requirement+.002,
                                              -item[1]),reverse=True)
                chosen_gains=[]
                for gain,_,_ in preview:
                    if all(abs(gain-other)>=.095 for other in chosen_gains):
                        chosen_gains.append(gain)
                    if len(chosen_gains)>=len(trial_gains):
                        break
            if pairwise_preview and len(used_modes)>=2:
                first,second=np.argsort(abs(coefficients*scale))[-2:]
                pair=[]
                bounds=[min(.2,max_step_rad/max(float(np.max(abs(used_modes[i]))),1e-12))
                        for i in (first,second)]
                for a in np.linspace(-1,1,9):
                    for b in np.linspace(-1,1,9):
                        if abs(a)+abs(b)<1e-12:
                            continue
                        ca,cb=a*bounds[0],b*bounds[1]
                        pair_step=ca*used_modes[first]+cb*used_modes[second]
                        if np.max(abs(pair_step))>max_step_rad+1e-12:
                            continue
                        predicted=baseline+ca*complex_columns[first]+cb*complex_columns[second]
                        predicted_rms,predicted_overlap=_metrics(
                            predicted,desired,weights,phase_valid,desired_intensity)
                        if (predicted_overlap>=shape_requirement+.001 and
                                predicted_rms<before_rms):
                            pair.append((predicted_rms,-predicted_overlap,
                                         ca,cb,pair_step))
                pair.sort(key=lambda item:(item[0],item[1]))
                pair_steps=[]
                for predicted_rms,_,ca,cb,pair_step in pair:
                    if any(np.sqrt(np.average((pair_step-other)**2,
                                               weights=weights))<.025
                           for _,other in pair_steps):
                        continue
                    pair_steps.append((f"response_pair_{ca:+.3f}_{cb:+.3f}",pair_step))
                    if len(pair_steps)>=len(trial_gains):
                        break
                if pair_steps:
                    trial_steps.extend(pair_steps)
                    checks.append({"kind":"pairwise_preview",
                                   "modes":[int(first),int(second)],
                                   "candidates":[name for name,_ in pair_steps]})
            checks.append({"kind":"fit","step_peak_rad":float(np.max(abs(step))),
                           "step_rms_rad":float(np.sqrt(np.mean(step**2))),
                           "predicted_phase_rms_rad":float(np.sqrt(np.sum(
                               sample_weight*fit_phase_prediction**2))),
                           "preview_selected_gains":chosen_gains})
            if not trial_steps:
                trial_steps.extend((f"response_fit_{gain:g}",gain*step)
                                   for gain in chosen_gains)
        except np.linalg.LinAlgError as exc:
            checks.append({"kind":"fit","passed":False,
                           "error":f"{type(exc).__name__}: {exc}"})
    trial_steps.extend((str(name),np.asarray(value,float))
                       for name,value in fallback_steps)
    for name,trial_step in trial_steps:
        if evaluations>=max_evaluations:
            break
        try:
            field,valid=simulate(np.mod(command+trial_step,2*np.pi))
            field=np.asarray(field,complex)
            if field.shape!=baseline.shape or not np.all(np.isfinite(field)) or not valid:
                raise ValueError("invalid physical verification result")
            rms,overlap=_metrics(field,desired,weights,phase_valid,desired_intensity)
            passed=(before_rms-rms>=min_phase_improvement_rad and
                    overlap>=shape_requirement)
            checks.append({"kind":"verification","candidate":name,
                           "phase_rms_rad":rms,"shape_overlap":overlap,
                           "phase_improvement_rad":before_rms-rms,
                           "passed":bool(passed)})
            if passed and (best is None or rms<best.phase_rms_rad):
                best=TeacherResult(trial_step.copy(),field,rms,overlap,[],0,name)
        except Exception as exc:
            checks.append({"kind":"verification","candidate":name,"passed":False,
                           "error":f"{type(exc).__name__}: {exc}"})
        evaluations+=1
    if best is None:
        return TeacherResult(None,None,None,None,checks,evaluations)
    best.checks=checks
    best.evaluations=evaluations
    return best
