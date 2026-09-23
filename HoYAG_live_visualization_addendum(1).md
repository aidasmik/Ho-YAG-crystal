# Ho:YAG live scientific visualization — agent implementation addendum

## Objective and scope

Extend the previously specified Ho:YAG externally seeded multipass amplifier with an interactive, live scientific visualization interface. Retain the selected optical geometry, material model, and disk–bond–cooling-plate assembly. Do not convert the amplifier into an oscillator for the visualization.

The interface must display the numerical solution, not generate a plausible-looking animation independently of the solver. This document is an implementation specification; it is not a functioning dashboard or evidence of achieved simulation performance.

The original YbSLAM proposal specifies a 10 kHz seed and 10 Hz phase-control updates (Application_Taiwan-1.pdf, pp. 11–12). These timings were retained in the Ho adaptation. They do not specify the rendering frame rate or the actual phase-modulator response time. The interface architecture, frame-rate targets, and preview modes below are proposed additions, not project specifications.

## 1. Keep three notions of real time separate

1. **Live reference mode:** show accepted states from the full numerical model while it runs. One simulated second may take more or less than one wall-clock second. The viewer must remain responsive even when solving is slow.
2. **Interactive preview mode:** use a demonstrably converged coarser model or a validated reduced-order model for parameter exploration. Display its approximation, validated parameter range, and error estimates. Never silently substitute it for the reference solver.
3. **Replay mode:** play recorded reference states, with pause, scrubbing, and speed control. Clearly label playback; it is not a live recalculation.

Target 30 rendered frames/s initially; benchmark rather than promise this throughput. This is a display target, not a requirement to solve the full multiphysics system 30 times/s. Show render FPS, accepted solver updates/s, simulated time, state age, and simulated-seconds/wall-second. Repeated display of the same state is allowed; inventing new physical states is not.

## 2. Required linked views

### A. Optical assembly and beam path

Show the seed, input SLM, beam-shaping optics, signal relays, repeated encounters with the SAME disk, separate pump-recycling path, diagnostic extraction, and output plane. Preserve coordinate frames and count disk encounters separately from material traversals.

Use beam centerlines for orientation only. Scientific beam textures and transverse slices must come from calculated complex fields. Distinguish pump and signal; label any colors as false color. A luminous tube is a diagrammatic overlay, not a prediction of what a camera sees in free space.

Allow selecting a component, disk encounter, or observation plane. In the first version, show transverse slices at actual solver planes. Only add a continuous longitudinal intensity volume after calculating enough propagation planes to support it.

A pulse-flight illustration may use explicitly slowed time. It must not share an unlabeled time axis with thermal transients, and must not pretend to resolve the optical carrier. Do not animate a vortex's intensity as rotating merely because its phase winds azimuthally.

### B. Incoming light

Show the complex-envelope-derived intensity or fluence map and phase at the seed and amplifier entrance. Specify field normalization, transverse pixel spacing, coordinate frame, wavelength, plane identifier, pulse identifier or averaging interval, and power/energy units.

If A is normalized so |A|^2 is W/m², use I=|A|^2. If the solver stores electric field in V/m, apply its physical intensity convention instead. For pulsed data, distinguish instantaneous or peak intensity (W/m²), fluence (J/m²), pulse energy (J), and time-averaged power (W). Never call all of these “intensity.”

Provide both an absolute-scale view and a separately labeled shape-normalized view. Do not independently normalize input and output without displaying their energies and amplification. Mask phase in pixels where the field is negligible.

### C. Phase modulator

Show the intended structured-light mask, correction mask, total requested mask, and actual applied phase separately:

    phi_requested = wrap_2pi(phi_pattern + phi_correction)
    A_after = a_SLM * A_before * exp(i * phi_applied)

For the ideal phase-only test, a_SLM=1: intensity immediately after the modulator must equal intensity immediately before it. Beam reshaping is revealed after propagation or filtering. Show this distinction in the UI. Account for diffraction-order selection and loss in the applicable propagation segment, not with an arbitrary brightness change.

Implement controller updates at the configured physical control period, independent of viewer FPS. For the retained 10 Hz setting this is 0.1 s. At 10 kHz, there are 1,000 seed pulses between command updates.

Model exposure, processing delay, command latency, phase quantization, wavelength calibration, and phase response only when parameters are available. Otherwise label the modulator and sensor as idealized. A fixed phase plate is static; only a modeled actuator/SLM receives time-varying commands. Hardware optical response must not be fabricated by interpolating wrapped phase colors.

### D. Crystal and cooling-plate temperature

Show T(x,y,z,t), front and rear disk surfaces, a through-thickness section, the bond/contact region, and plate temperature. Include a separate |grad(T)| map with K/mm or K/m units. A temperature map alone is not a gradient map.

Show heat-flux vectors or streamlines based on q=-k grad(T), using the thermal-conductivity tensor when applicable. Label the vector scaling and distinguish heat-flow streamlines from coolant-fluid streamlines.

Plot maximum disk temperature, spatial temperature variation over an explicitly defined region, plate temperature, and temperature jump across the contact. Use controlled/common color limits for comparisons; expose any autoscaling.

### E. Disk bending and mechanical stress

Render displacement from the mechanical solution, not an imposed bowl proportional to pump power. Show the original geometry as an optional wireframe and the deformed assembly as a separate surface. Include a displacement/height map in nm or µm, disk thickness change, and relevant stress measures.

The displayed geometry may be:

    x_display = x_reference + exaggeration_factor * u_physical

Always display the factor, such as “deformation ×1000.” Never feed exaggerated geometry back to optics, heat transfer, contact, or mechanics. Keep raw displacement distinct from a detrended surface-figure map; state whether piston and tilt have been removed.

Mechanical boundary conditions must represent the crystal, bond/contact and mount/plate. Uniform unconstrained heating should produce expansion, not an invented bending mode. Asymmetric illumination, nonuniform doping/contact, or misalignment must not be forced into an axisymmetric solution. A reduced axisymmetric model is allowed only for configurations that satisfy its symmetry assumptions.

Use quasistatic thermoelasticity only within its applicable time regime. Claims about acoustic ringing or pulse-driven mechanical vibration require a transient mechanical model; do not animate such effects using a quasistatic solution.

### F. Cooling performance and capacity

Display, separately:

- heat deposited in the disk and coatings, in W;
- heat transferred from disk to plate, in W;
- heat transferred from the modeled assembly into coolant, in W;
- thermal-energy storage rate in the modeled solids, in W;
- coolant inlet/outlet temperatures and mass flow, only when modeled;
- available cooling capacity/headroom, only when a valid limit or chiller performance model is supplied.

Compute actual boundary heat removal from the signed flux integral, with the outward normal pointing from the solids toward the coolant:

    P_to_coolant = integral_over_coolant_boundary(q dot n dA)

At a finite thermal contact use a physically specified conductance, for example q_contact=h_contact*(T_disk-T_plate). Contact conductance is not the same parameter as coolant-side convective heat-transfer coefficient.

A prescribed coolant temperature is an ideal boundary condition, not proof of finite chiller capacity. Without a specified capacity curve/limit, show capacity as “not modeled,” not a made-up utilization percentage. Do not clip boundary heat flux at a nominal limit while maintaining an incompatible fixed coolant temperature: a capacity-limited system requires a consistent coolant/reservoir/chiller energy balance.

The familiar steady-flow coolant relation P=mass_flow*cp*(T_out-T_in) is a consistency check only when its assumptions apply. Include fluid thermal storage and transport delay for transient coolant modeling; do not force instantaneous inlet/outlet calorimetry to match the solid-side flux.

Monitor the solid thermal balance:

    dU_thermal/dt = P_heat_deposited - P_to_coolant - P_other_boundary_heat_losses

Also retain a separate optical/population energy balance. Absorbed pump power is not automatically all heat: optical extraction, fluorescence, changing stored excitation, and nonradiative channels must follow the Ho material model.

### G. Output and correction performance

Show near-field and selected far-field/focal-plane intensity or fluence, phase, output pulse energy, average power, per-pass gain/loss, wavefront error, and a clearly defined target-mode overlap. Use the actual calculated output; do not enforce any project energy or gain target in the display.

Compare cold/reference, heated uncorrected, and heated corrected runs at matching operating conditions and times. Keep energy scales and phase conventions consistent. A vortex must not be evaluated against a flat-phase Gaussian target. Define the comparison plane, normalization, mask, and any removed global phase.

Separate the exact simulated phase (“ground truth”) from a synthetic camera/interferometer measurement. Do not claim that an intensity-only detector directly measures phase. Phase recovery must use the modeled measurement arrangement or remain explicitly idealized.

## 3. Physics-to-visualization contract

The causal model is:

    pump transport + Ho populations + signal propagation
        -> deposited heat
        -> transient disk/contact/plate temperature
        -> thermoelastic displacement and stress
        -> optical-path and gain updates
        -> propagated output
        -> synthetic measurement/controller
        -> future input SLM command

Temperature-dependent index, mechanical displacement, and photoelastic terms must use consistent definitions. Obtain optical phase from the physical ray/field path and interface locations. Do not count the same expansion twice through both a moved boundary and a duplicate path-length term. A simple mirror-displacement phase factor is not automatically the complete optical-path change through a refracting, double-pass crystal.

All disk visits share the same physical populations, material maps and thermal/mechanical state, with registered coordinates and correct visit timing. Do not reset inversion, regenerate the pump, or independently bend copies of the disk between passes.

Use optical-envelope propagation and pulse/population updates appropriate to the model. Solve thermal and mechanical evolution on justified slower time scales. Pulse averaging or multi-pulse jumps require comparison with resolved reference cases, especially during startup, pump changes and SLM updates. Do not replace a saturating pulse train with a CW signal merely to achieve a smooth display.

## 4. Software structure

Retain existing validated numerical kernels. Put solver execution in a worker/process separate from rendering. The frontend sends timestamped control requests; it does not mutate solver arrays while they are in use.

Publish immutable, versioned snapshots after accepted solver updates. Each snapshot should contain:

    run_id, configuration_hash, state_id
    t_sim_s, t_published_wall, time_interval_or_pulse_id
    fidelity_mode, solver_status, convergence_diagnostics
    mesh_ids, coordinate_frames, units
    optical_fields_at_selected_planes
    pump_absorption_and_population_fields
    phi_pattern, phi_correction, phi_requested, phi_applied
    T_disk, T_plate, grad_T, heat_flux
    displacement, relevant_stress_fields
    optical_path_error
    energy_power_cooling_metrics
    approximation_flags, sensor_model_flags

If subsolvers update at different times, publish a synchronized accepted macrostate or expose the timestamps of held/interpolated components. Never silently combine a new optical field with an unrelated old temperature or command.

Use bounded queues and allow the viewer to skip old display frames without skipping required physical integration events. Do not increase numerical time steps simply because the viewer is slow. Record physical parameter changes and their effective simulated time. Geometry/material changes may require restarting a run; show that explicitly.

A suitable initial interface is PyVista/VTK for meshes and field views, with trame for browser controls/remote visualization. FEniCSx is an option for missing thermoelastic components, not a requirement to rewrite a working solver. Larger parallel runs may later use ParaView/Catalyst. These tools provide visualization/coupling infrastructure, not automatic Ho:YAG physics or guaranteed real-time throughput.

Persist scientific data independently of screenshots. Save raw fields or selected checkpoints, units, configuration, event log, solver tolerances, and comparison metrics. A video alone is not a reproducible scientific result.

## 5. Initial controls

Provide pump power and pump position, seed energy, selected structured-light mask, phase-control enable/disable, supported coolant conditions, and a validated contact-conductance parameter. Expose disk thickness or pass count as restart-required configuration changes when necessary.

Provide pause/resume, single solver step, selected pass/plane, slicing and camera controls, fixed/automatic color limits, deformation magnification, raw versus synthetic measurement, recording, and reference/preview/replay mode.

Do not offer a “cooling capacity” slider unless it changes a defined capacity-limited cooling model. Do not offer coolant-flow controls that merely recolor an assumed temperature field.

## 6. Implementation order and acceptance checks

**First:** build a viewer/replay adapter for accepted solver outputs. This checks units, coordinates, pass identity, and field-to-geometry mapping without changing numerical physics.

**Second:** connect it to a running worker, add synchronized state publication and timestamped controls, and benchmark solver speed separately from rendering speed.

**Third:** add reduced-order preview only after comparing it against the reference solver over a declared operating range.

Required validation includes pure-phase local intensity preservation; passive propagation energy accounting including physical losses; pump-off thermal relaxation; contact heat-flow continuity; thermal energy balance; unconstrained thermal expansion; correct physical displacement in optics; and consistent results when the viewer is disabled or its FPS changes. Verify optical-grid, thermal/mechanical-mesh, time-step, and pulse-averaging convergence for the reported metrics.

Demonstrate a pump step, a reduced-cooling case, and correction enable/disable with solver-driven transients. Do not prescribe the direction or size of output improvement: the chosen configuration or controller may fail, and that result must remain visible.

Report tested hardware, mesh sizes, optical grid, model assumptions, solver update rate, render FPS, latency, and real-time factor. Missing material, bonding, mounting, actuator or cooling data must remain explicit uncertainties. Numerical verification is not experimental validation.

## Supporting sources

These support the implementation methods, not Ho:YAG parameter values or promised performance.

1. Application_Taiwan-1.pdf, pp. 11–12: original seed, monitoring and active-control specifications. The Ho adaptation is separate from the Yb-based proposal.
2. Kitware, *Monitor your Simulation in your Web Browser with trame and Catalyst*: `https://www.kitware.com/monitor-your-simulation-in-your-web-browser-with-trame-and-catalyst/`
3. PyVista documentation, *Trame Jupyter Backend*: `https://docs.pyvista.org/user-guide/jupyter/trame.html`
4. LightPipes manual, field propagation and phase/intensity operations: `https://opticspy.github.io/lightpipes/manual.html`
5. J. Bleyer, *Linear thermoelasticity (weak coupling)*, FEniCSx numerical tour: `https://bleyerj.github.io/comet-fenicsx/tours/linear_problems/thermoelasticity_weak/thermoelasticity_weak.html`
6. COMSOL documentation, *Thermal Contact*: `https://doc.comsol.com/6.3/doc/com.comsol.help.heat/heat_ug_ht_features.09.093.html`

## Final instruction to the implementation agent

Build a live, solver-driven scientific interface, not a demonstration with pre-scripted temperature, bending, gain, or beam changes. Prefer honest slower-than-real-time reference results over an apparently real-time display with undisclosed approximations. Show what was calculated, when it was calculated, and the assumptions under which it is valid.