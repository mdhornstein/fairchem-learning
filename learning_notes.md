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