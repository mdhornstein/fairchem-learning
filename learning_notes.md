## Concept Check 0 (The Scientific Motivation)

Question: In quantum chemistry and materials science, what fundamental physical equation or quantities are traditional methods like Density Functional Theory (DFT) solving for, and what is the computational bottleneck that motivates machine learning frameworks like FAIRChem?

## Concept Check 1 

If the machine learning model calculates the potential energy ($E$) of a molecule at a given 3D arrangement, how does it mathematically figure out the physical force ($\vec{F}$) pushing on each atom? (Hint: Think about high school physics—like a ball sitting on a hill. How does the slope of the hill relate to the direction the ball wants to roll?)


force is probably negative gradient of potential energy with respect to that atom's coordinates in space 

## Concept Check 2  (The Geometry of Chemistry)

What is the fundamental difference between simulating an isolated molecule (like a single caffeine molecule in a vacuum) versus a bulk material / crystal (like solid iron, a battery cathode, or a catalyst surface)?

How do computational chemists simulate an infinite bulk material on a computer without having to simulate an infinite number of atoms?


maybe they simulate what happens to the unit cell and the atoms immediately surrounding the unit cell 

## Concept Check 3 (The Symmetry Problem)

A standard image neural network (CNN) takes a fixed grid of pixels. But a molecule in 3D space is different:

What happens to the true physical energy of that water molecule if you rotate it by $45^\circ$ in 3D space?
What happens if you swap the order in Python list from ["O", "H", "H"] to ["H", "O", "H"]?
Does a standard Multi-Layer Perceptron (MLP) or simple feedforward network naturally respect those physical rules?

## Concept Check 4 (Periodic Systems)

If two copper atoms are in a cubic unit cell of length $10\text{ Å}$, Atom A is at $(1, 0, 0)$ and Atom B is at $(9, 0, 0)$... In an isolated molecule (pbc=False), the distance between them is $8\text{ Å}$. In a periodic crystal (pbc=True), what is the true shortest physical distance between them through the boundary wall?

## Concept Check 5 (Moving from Atoms to Graphs)

In a real crystal or surface catalyst, periodic images repeat infinitely in 3D space.

When FAIRChem converts this system of atoms into a Graph Neural Network (GNN):

What are the nodes of the graph?
What are the edges of the graph?
Why can't we draw an edge between every atom and every other atom? How do we decide which atoms get connected by an edge?


1. Bonds form and break dynamically; at what exact picosecond does a bond break? [arbitrary?]
2. metals don't have discrete bonds; instead delocalized sea of electrons 
3. non-bond interactions also matter 

spatial cutoff graph 
* choose a physical cutoff radius, like 6 Angstroms 
* If two atoms are closer then the cutoff, we draw an edge 
* Interactions decay with distance 
* Without a cutoff, compute and memory scale as $O(N^2)$, but with a cutoff, they scale as $O(N)$ 
* [Seems like this is a difference between molecules and sentences; with sentences, the scaling is N^2, because there is no natural "cutoff" you can define.]

## Concept Check 6 (Direction vs. Distance)

When an edge connects Atom $i$ and Atom $j$:

The scalar distance is $d_{ij} = |\vec{r}_j - \vec{r}_i|$.
The relative displacement vector is $\vec{r}_{ij} = \vec{r}_j - \vec{r}_i$ (which has both distance and 3D direction in space).
If our AI model only looked at the scalar distances $d_{ij}$ and ignored the 3D direction vectors $\vec{r}_{ij}$, could it correctly predict the 3D directional vectors of the forces ($\vec{F}_x, \vec{F}_y, \vec{F}_z$) pushing on the atoms? Why or why not?



## Concept Check 7 (Invariance vs. Equivariance)

Imagine a hot cup of coffee sitting on your desk. You pick up the cup and rotate it $90^\circ$ clockwise.

Name one physical property of the coffee that is invariant (stays completely unchanged).
Name one physical property of the coffee (or its handle/molecules) that is equivariant (changes, but in exact lockstep with your rotation).

## Concept Check 8 (Message Passing)

How does a Graph Neural Network (GNN) actually compute properties using this graph?

The core algorithm is called Message Passing. In plain English: If Atom $i$ and Atom $j$ are neighbors connected by an edge, what kind of "message" does Atom $i$ send to Atom $j$, and what does Atom $j$ do with all the messages it receives from its neighbors?


I suppose a message is an MLP layer or something like that, i.e. node i outputs a vector, and it is fed through an MLP to become an input to node j. But a given node may have multiple inputs, so they would need to be combined somehow. 


## Concept Check 9 (Batching Variable-Sized Graphs)


In standard Computer Vision, 16 images in a batch are stacked into a 4D tensor: (16, 3, 224, 224) because every image has the same number of pixels.

In chemistry:

Molecule A has 3 atoms (Water: $H_2O$)

Molecule B has 24 atoms (Caffeine)

Molecule C has 80 atoms (a catalyst surface)

Why can't we simply stack them into a standard 3D tensor like (batch_size, num_atoms, 3)?
How might a Graph Neural Network batch multiple different molecules into a single giant graph for the GPU?

1. We can't simply batch them because the dimensions don't match up. 
2. Ragged tensor? Only batch molecules of the same shape?  Aha!  Take the initial molecules, and embed them into a common dimension. Not sure how to do this embedding. If each atom is represented by a vector, are we batching atoms or molecules? Seems like we have multiple hierarchical layers, atoms are organied into molecules, and molecules are organized into batches of molecules? 



## Concept Check 10 (The Potential Energy Surface & Relaxation)

In computational chemistry and materials science:

What is the Potential Energy Surface (PES)?
If you place a reactant molecule onto a catalyst surface in an initial rough guess, the atoms will experience strong forces ($\vec{F} \neq 0$). What computational procedure do scientists run to find the stable, resting equilibrium shape of the system? (Hint: What happens if you iteratively move each atom in the direction its force is pushing until the forces drop close to zero?)


PES is probably the potential energy as a function of the configuration of the molecule(s) (i.e. the position of each of the atoms). To find the equilibrium, you can simulate small timesteps with little increments of motion according to the forces at each timestep, until the system reaches equilibrium (or keeps oscillating?)



## Concept Check 11 ("Conserving" vs. "Direct" Models)
Look closely at two of the model names printed by that list:

esen-sm-conserving-all-omol
esen-sm-direct-all-omol
Connecting back to our very first discussion about PyTorch and forces:

What is the mathematical and architectural difference between a conserving potential and a direct force prediction model? (Hint: How is force computed in each one, and what is the trade-off in compute speed vs physical guarantees?)



I don't know what a conserving potential is. Does it predict the final relaxed state as a one-off? If so, it has lower compute requirements (just a single one-off prediction rather than predicting the force at each timestep), but worse performance on OOD (less flexibility to model new configurations, because it doesn't model the mechanism, just the outcome) 




## Concept Check 12 (Energy Drift)

If you run a 1,000,000-step Molecular Dynamics simulation of a vibrating crystal using a Direct model whose forces have a tiny $0.01%$ non-conservative bias at every step:

What will happen to the total energy and temperature of the crystal over the course of those 1,000,000 steps?


The total energy will likely drift over time. Don't know what will happen to the temperature; maybe fluctuate randomly. 


You got the drift part right! And here is the punchline for temperature: In physics, temperature is literally just the average kinetic energy of the atoms ($E_{\text{kinetic}} = \frac{3}{2} k_B T$).

If a non-conservative model continuously injects a tiny fraction of phantom energy at each step, that energy turns into kinetic motion: the atoms vibrate faster and faster, the temperature shoots up uncontrollably, and within a few thousand steps your solid crystal spontaneously boils and explodes into gas! That's why conserving models are essential for stable Molecular Dynamics.



Why does the energy become kinetic energy as opposed to potential energy? And why does the energy necessarily go up instead of down? 



1. Two Brilliant Physics Questions Answered
In your notes, you asked two profound questions:

Q1: "Why does the energy become kinetic energy as opposed to potential energy?"
Answer: It becomes both! In a molecule or crystal, atoms are connected by chemical bonds that act like quantum springs. When an atom vibrates, kinetic energy and potential energy continuously slosh back and forth. By the Virial Theorem / Equipartition Theorem, in any vibrating system, energy partitions roughly 50% into potential energy and 50% into kinetic energy on average. Because temperature is defined strictly by kinetic energy ($\langle E_k \rangle = \frac{3}{2} k_B T$), any increase in total energy inevitably increases the kinetic energy, driving the temperature up!

Q2: "Why does the energy necessarily go UP (heating) instead of DOWN (cooling)?"
Answer: Because of the "Asymmetric Repulsive Wall." When two atoms get too close, their electron clouds overlap and Pauli exclusion kicks in, creating a ferociously steep repulsive wall (scaling as $1/r^{12}$ or exponential). If a numerical simulation or an imperfect model takes a step that accidentally pushes two atoms a tiny fraction of an Ångström too close into that wall, the repulsive force spikes exponentially. It violently "kicks" the atoms outward. That asymmetric kick does huge net positive work on the atoms. Over thousands of steps, these non-conservative kicks act like an invisible blowtorch pumping heat into the crystal until it blows apart.



So numerical error is essentially doing work, haha! 




                  ┌────────────────────────┐
                  │  Atoms / AtomicData    │
                  │ (pos, elements, cell)  │
                  └───────────┬────────────┘
                              │
                    [ GNN Backbone ]
              (Equivariant Message Passing)
                              │
                    Per-atom Embeddings
                   h_i (scalars & vectors)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     [ Energy Head ]                  [ Direct Force Head ]
   Scalar pooling: E = Σ e_i            Vector output: F_i ∈ R^3
              │                               │
       Autograd: -∇_r E                       │
              │                               │
              ▼                               ▼
    Conserving Forces                 Direct Forces





## Concept Check 13

In standard deep learning (like classifying an image of a dog or generating text with an LLM), we never set requires_grad=True on the input image or input text tokens. We only calculate gradients with respect to the weights ($W$).

Why do atomistic potentials like FAIRChem set pos.requires_grad = True on the input coordinates during inference?


pos stores the positions (coordinates) of the atoms. We need to differentiate the potential energy with respect to the coordinates in order to find the force. That's why we need to set requires_grad = True. 