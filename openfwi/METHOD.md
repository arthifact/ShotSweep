# Our method: seeing the subsurface from every shot

The first version of our pipeline read the five shot recordings like a flat-layer processor would: it picked slopes, voted for flat reflectors and handed the network a layered guess. That worked on flat geology and broke on folds, faults and textured models. The second version stops guessing the geology for the network. It gives the network what the physics can see, from every shot and under several velocity hypotheses, and lets the network decide.

## The training and architecture

Everything runs on a single GPU: the wave simulations that make the images and the network that learns from them. It was developed on an Apple M4 Pro laptop (Apple GPU) and runs unchanged on a CUDA workstation, so we control the data, the physics and the network end to end.

To improve the network's guesses, we combine three ideas.

### 1. Multi-view awareness

A single shot sees the subsurface from one side only. Instead of stacking the five shots as anonymous channels, the network treats them as five views of the same ground. One shared encoder reads each shot, and the views are fused so that every shot is handled the same way. Where the shots agree, the network can trust what it sees. Where they disagree, the velocity guess behind the image is wrong, and that disagreement is itself the signal the network learns from.

### 2. Rich pixel input

Raw recordings live in time and receiver position, not in the subsurface. So for every shot we back-propagate the recording through the wave equation into the subsurface, producing an image on the same 70 by 70 grid as the answer. Each pixel then carries:

* **Reflector identity:** whether a boundary between rock layers passes through this pixel, and how strong it is.
* **Polarity:** whether velocity steps up or down across that boundary, which is what reveals slow layers hidden under fast ones.
* **Velocity cues:** the same image computed under three background velocities, slow, medium and fast. A reflector that sits still across the five shots under one background, and drifts apart under the others, tells the network which velocity is right at that pixel.
* **Depth trend:** the background velocity itself, measured from the direct wave at the surface, so the network knows the large-scale increase of velocity with depth.

The raw recordings stay as a second input, so nothing the images miss is lost.

### Conditioning the loss function

Rather than relying only on pixel-by-pixel error, we shape what the network pays attention to:

* **Structural similarity (MS-SSIM):** pushes the training to get interfaces, faults and layer shapes right, not just average velocities. Published tests show it gives sharper velocity maps than pixel losses alone.
* **Physics consistency:** after the network predicts a model, we simulate the recordings it would produce and compare them with the real ones. When the two agree to within half a wave period, a few physics refinement steps finish the job. When they do not, we keep the network's answer instead of letting the physics drift into a wrong solution.

### Robustness

During training the image branch is randomly switched off part of the time, so the network never relies blindly on one input. Models and recordings are mirrored together, and all ten OpenFWI geology families are trained at once, the single strongest generalization gain reported on this benchmark.

## What it is built to do

By feeding the network images from every shot under several velocity hypotheses, and by checking its answers against the physics, the method is designed to recover curved layers, faults and textured geology that defeat flat-layer processing, and to learn from far fewer training models than networks that see only raw recordings. We will measure that against the same network on raw recordings and against InversionNet retrained on identical data, family by family, with the standard OpenFWI metrics.
