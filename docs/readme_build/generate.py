"""Build root README figures from the exact audited Stage 7 Actions artifact."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hoyag.propagation import Grid2D, laguerre_gaussian
from hoyag.populations import HoYAGFourLevelParams, I7, I8
from hoyag.resonator import ThinDiskResonator
from hoyag.thermal import DiskThermalMesh
from hoyag.coupled_resonator import PlateAssembly
from hoyag.vector_cavity import PlaneExchange, VectorRoundTrip, normalize_vector
from hoyag.thermomechanics import von_mises

ROOT=Path(__file__).resolve().parents[2]

def locate(root,name):
    found=list(root.rglob(name))
    if len(found)!=1:
        raise RuntimeError(f"Expected one {name}, found {found}")
    return found[0]

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifact-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=ROOT/"docs/results_readme")
    args=ap.parse_args()
    out=args.output; figs=out/"figures"; data=out/"data"
    figs.mkdir(parents=True,exist_ok=True); data.mkdir(exist_ok=True)

    state_p=locate(args.artifact_root,"state.npz")
    summary_p=locate(args.artifact_root,"summary.json")
    history_p=locate(args.artifact_root,"history.csv")
    verify_p=locate(args.artifact_root,"verification.json")
    shutil.copy2(summary_p,data/"stage7_summary.json")
    shutil.copy2(history_p,data/"stage7_history.csv")
    shutil.copy2(verify_p,data/"audit_verification.json")

    S=json.loads(summary_p.read_text())
    A=np.load(state_p,allow_pickle=False)
    x=A["x_m"]; y=A["y_m"]; dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    grid=Grid2D(nx=len(x),ny=len(y),dx=dx,dy=dy)
    xx,yy=grid.mesh; rr=np.hypot(xx,yy)
    cavity_meta=S["metadata"]["cavity"]
    cavity_keys=("disk_diameter_m","disk_thickness_m","air_gap_m","output_mirror_radius_m",
                 "output_transmission","disk_hr_reflectivity","other_roundtrip_loss","wavelength_m",
                 "host_index","host_group_index","pump_hr_reflectivity")
    c=ThinDiskResonator(**{k:cavity_meta[k] for k in cavity_keys})
    mesh=DiskThermalMesh(A["r_edges_m"],A["z_edges_m"],A["raw_heat_W_m3"].shape[-1])
    assembly=PlateAssembly(mesh,grid,S["assembly_configuration"])
    temp,disp,screens=assembly.solve(A["assembly_heat_W_m3"])
    exchange=PlaneExchange(grid,mesh,order=6)

    p=HoYAGFourLevelParams()
    f=A["mean_fractions"].reshape(4,mesh.nz,mesh.nr,mesh.nphi)
    gain=p.N_total_m3*(p.sigma_em_laser_m2*f[I7]-p.sigma_abs_laser_m2*f[I8])
    log_gain=exchange.surface_on_grid(np.sum(gain*np.diff(mesh.z_edges_m)[:,None,None],axis=0))
    operator=VectorRoundTrip(grid,c,screens,log_gain)
    field=normalize_vector(A["fields_used"][0])
    returned,oc,forward,backward=operator.propagate(field,return_visits=True)

    Pout=float(S["energy_budget"]["output_W"])
    Iout=np.sum(np.abs(oc)**2,axis=0)
    Iout*=Pout/(Iout.sum()*dx*dy)
    pump_w=float(S["metadata"]["pump_waist_m"])
    Ipump=2*S["energy_budget"]["pump_incident_W"]/(np.pi*pump_w**2)*np.exp(-2*rr**2/pump_w**2)

    def surface(values):
        return exchange.surface_on_grid(values)

    n7=surface(np.mean(f[I7],axis=0))
    heat=surface(np.mean(A["raw_heat_W_m3"],axis=0))
    Tfront=surface(temp.disk_temperature_K[0])
    mask=rr<=c.disk_diameter_m/2
    manifest=[]

    def save(name,desc):
        plt.tight_layout()
        path=figs/f"{name}.png"
        plt.savefig(path,dpi=180,bbox_inches="tight")
        plt.close()
        manifest.append({"file":path.name,"description":desc})

    def im(name,z,title,label,extent=1.5,cmap=None,vmin=None,vmax=None):
        plt.figure(figsize=(6.3,5))
        h=plt.pcolormesh(x*1e3,y*1e3,np.ma.masked_where(~mask,z),shading="nearest",
                         cmap=cmap,vmin=vmin,vmax=vmax)
        plt.colorbar(h,label=label); plt.xlabel("x (mm)"); plt.ylabel("y (mm)")
        plt.xlim(-extent,extent); plt.ylim(-extent,extent); plt.gca().set_aspect("equal")
        plt.title(title); save(name,title)

    # 1 cavity
    z=np.linspace(0,c.disk_thickness_m+c.air_gap_m,700)
    red=np.where(z<c.disk_thickness_m,z/c.host_index,c.disk_thickness_m/c.host_index+z-c.disk_thickness_m)
    zr=c.rayleigh_range_m; w=c.waist_m*np.sqrt(1+(red/zr)**2)
    plt.figure(figsize=(8.6,4.3)); plt.plot(z*1e3,w*1e3); plt.plot(z*1e3,-w*1e3)
    plt.axvspan(0,1,alpha=.18,label="1 mm Ho:YAG"); plt.axvline(z[-1]*1e3,ls="--",label="2% output coupler")
    plt.xlabel("Distance from rear HR (mm)"); plt.ylabel("1/e² radius (mm)")
    plt.title("250 mm / 2% thin-disk reference resonator"); plt.legend()
    save("01_resonator","Cold-cavity design envelope")

    # 2 uniform Ho
    im("02_ho_distribution",np.where(mask,p.N_total_m3/1e26,np.nan),
       "Total Ho density used by the coupled reference","NHo (10²⁶ m⁻³)",extent=5.3,vmin=0,vmax=1.6)

    # 3 pump
    im("03_pump_input",Ipump/1e4,"Incident pump | 1907.7 nm, 10 W average","W/cm²",extent=1.5,vmin=0)

    # 4 disk mode
    Idisk=np.sum(np.abs(field)**2,axis=0); Idisk/=Idisk.max()
    im("04_cavity_mode",Idisk,"Converged Stage 7 vector mode at disk","Intensity / peak",extent=1.5,vmin=0,vmax=1)

    # 5 output
    im("05_output_beam",Iout/1e4,"Useful output at output coupler | 2090.3 nm","Average intensity (W/cm²)",extent=1.5,vmin=0)
    phase=np.angle(oc[0]); phase=np.ma.masked_where(np.abs(oc[0])**2<1e-4*np.max(np.abs(oc[0])**2),phase)
    plt.figure(figsize=(6.3,5)); h=plt.pcolormesh(x*1e3,y*1e3,phase,shading="nearest",cmap="twilight",vmin=-np.pi,vmax=np.pi)
    plt.colorbar(h,label="Wrapped phase (rad)"); plt.xlim(-1.5,1.5); plt.ylim(-1.5,1.5); plt.gca().set_aspect("equal")
    plt.xlabel("x (mm)"); plt.ylabel("y (mm)"); plt.title("Output phase | laboratory x polarization")
    save("06_output_phase","Global phase arbitrary; low-intensity pixels masked")

    # 7 excitation
    im("07_upper_manifold",100*n7,"Cycle-averaged upper laser manifold","N7 / NHo (%)",extent=1.8,vmin=0)

    # 8 heat
    im("08_heat_source",heat/1e9,"Thickness-averaged deposited heat","GW/m³",extent=1.8,vmin=0)

    # 9 temp
    im("09_disk_temperature",Tfront-273.15,"Crystal temperature | front-adjacent cell layer","°C",extent=1.8)

    # 10 plate/crystal section
    plate=assembly.heat_solver.plate
    plt.figure(figsize=(8,4.4))
    vmax=max(temp.disk_temperature_K.max(),temp.plate_temperature_K.max())-273.15
    plt.pcolormesh(mesh.r_edges_m*1e3,mesh.z_edges_m*1e3,temp.disk_temperature_K[:,:,0]-273.15,
                   shading="flat",vmin=20,vmax=vmax)
    h=plt.pcolormesh(plate.r_edges_m*1e3,(plate.z_edges_m+mesh.z_edges_m[-1])*1e3,
                     temp.plate_temperature_K[:,:,0]-273.15,shading="flat",vmin=20,vmax=vmax)
    plt.colorbar(h,label="°C"); plt.axhline(1,ls="--",label="crystal / plate interface")
    plt.gca().invert_yaxis(); plt.xlabel("radius (mm)"); plt.ylabel("depth (mm)")
    plt.title("Finite Ho:YAG + copper cooling plate temperature"); plt.legend()
    save("10_assembly_temperature","Recomputed from archived final relaxed Stage 7 heat")

    # 11 deformation
    im("11_front_deformation",screens.front_uz_m*1e9,"Thermo-mechanical crystal front displacement","u_z (nm)",extent=1.5,cmap="coolwarm")
    im("12_rear_deformation",screens.rear_uz_m*1e9,"Thermo-mechanical crystal rear-HR displacement","u_z (nm)",extent=1.5,cmap="coolwarm")

    # 13 OPD
    opd=screens.mean_roundtrip_opd_m
    beam=rr<=1.2e-3; piston=float(np.mean(opd[beam]))
    im("13_hot_disk_opd",(opd-piston)*1e9,"Total mean hot-disk round-trip OPD","nm relative to central-region piston",extent=1.5,cmap="coolwarm")

    # 14 convergence
    hist=S["history"]; it=[r["iteration"] for r in hist]
    plt.figure(figsize=(7.2,4.6))
    plt.semilogy(it,[r["field_residual"] for r in hist],"o-",label="field")
    plt.semilogy(it,[np.nan if r["heat_residual"] is None else r["heat_residual"] for r in hist],"s-",label="raw heat")
    plt.xlabel("outer iteration"); plt.ylabel("fixed-point residual"); plt.title("Coupled hot-cavity convergence")
    plt.legend(); save("14_convergence","Fixed-point convergence, not mesh convergence")

    # 15 waveform
    wave=A["optical_waveform"]
    plt.figure(figsize=(7.2,4.6)); plt.plot((wave[:,0]-wave[:,0].min())*1e6,wave[:,1])
    plt.xlabel("time in archived captured cycle (µs)"); plt.ylabel("output power (W)")
    plt.title("Archived laser output waveform"); save("15_output_waveform","10 ps pump does not imply 10 ps oscillator output")

    # 16 seeded LG weak diagnostic
    lg=laguerre_gaussian(grid,0,1,c.waist_m); v=normalize_vector(np.stack([lg,np.zeros_like(lg)]))
    _,lgout=operator.propagate(v)
    lin=np.sum(np.abs(v)**2,axis=0); lout=np.sum(np.abs(lgout)**2,axis=0)
    scale=max(lin.max(),lout.max())
    im("16_lg1_input",lin/scale,"Seeded LG₀¹ diagnostic | input at disk","common normalized intensity",extent=1.5,vmin=0,vmax=1)
    im("17_lg1_output",lout/scale,"Seeded LG₀¹ diagnostic | output after one hot round trip","common normalized intensity",extent=1.5,vmin=0,vmax=1)

    metrics={
      "source_artifact_state_sha256":sha256(state_p),
      "source_artifact_summary_sha256":sha256(summary_p),
      "output_W":Pout,
      "absorbed_pump_W":float(S["energy_budget"]["pump_absorbed_W"]),
      "heat_W":float(S["energy_budget"]["heat_W"]),
      "peak_disk_temperature_K":float(temp.disk_temperature_K.max()),
      "peak_plate_temperature_K":float(temp.plate_temperature_K.max()),
      "front_displacement_pv_nm_within_1p5mm":float(np.ptp(screens.front_uz_m[rr<=1.5e-3])*1e9),
      "rear_displacement_pv_nm_within_1p5mm":float(np.ptp(screens.rear_uz_m[rr<=1.5e-3])*1e9),
      "roundtrip_opd_pv_nm_within_1p5mm":float(np.ptp(opd[rr<=1.5e-3])*1e9),
      "stage7_converged":bool(S["converged"]),
      "outer_iterations":int(S["outer_iterations"]),
      "note":"Figures are post-processing of the audited Stage 7 artifact. No new nonlinear oscillator fixed point is solved by this documentation build."
    }
    (data/"derived_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (figs/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    A.close()
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
