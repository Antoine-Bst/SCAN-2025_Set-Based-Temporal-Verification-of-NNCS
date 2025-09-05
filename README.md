Welcome ! This is a prototype for the verification of STL formula on reachable tube of Neural Network Controlled System using Dynibex. It has been tested on Linux Ubuntu only.
It uses DynIbex.

Install Dynibex:
https://perso.ensta-paris.fr/~chapoutot/dynibex/index.php#download-installation

In recent Ubuntu versions you might need to do the the install in a python 2.7 virtual environement and :
sudo CXXFLAGS="-std=c++14" ./waf configure
sudo CXXFLAGS="-std=c++14" ./waf install

If dynibex is a local install add to the make file:
export PKG_CONFIG_PATH='path_to_dynibex'/share/pkgconfig 

Make the code:
Open terminal in the directory and compile using make:
make

Run to compute the reachable tube of a simple robot navigating in a grid of motion primitives:
./simulation.out

A small example with a NonLinear Car model inspired from Dubins model is provided
make -> ./simulation.out

if it runs:
---------------Sequence : 3---------------
---------------Noeud : 0---------------
Layer 1: 40 x 3
Layer 2: 20 x 40
Layer 3: 2 x 20
Processing simulation 0 -> result/predicate_3.000.txt, result/jn_box_3.000.txt
Files saved: result/predicate_3.000.txt, result/jn_box_3.000.txt
---------------Sequence : 4---------------
---------------Noeud : 0---------------
Layer 1: 40 x 3
Layer 2: 20 x 40
Layer 3: 2 x 20
should appears in the terminal as well as reachable sets in /result

=====================================Small Documentation===================================
The verification is performed in a bottom-up manner using the syntax tree of the formula and the satisfaction signals

phi1 = neg_stl(phi);             // Logical negation: ¬phi
phi1 = and_stl(phi2, phi3);      // Logical AND: phi2 ∧ phi3
phi1 = or_stl(phi2, phi3);       // Logical OR: phi2 ∨ phi3
phi1 = until_stl(phi2, phi3, {t1, t2});  // Until operator: phi2 U[t1,t2] phi3
phi1 = Finally(phi, {t1, t2});   // Eventually operator: F[t1,t2] phi
phi1 = Globally(phi, {t1, t2});  // Always operator: G[t1,t2] phi


predicate_satisfaction(sim, predicates); Constructs satisfaction signals for the list of predicates and the simulation object.
The output is a list of signals corresponding to each predicate in the input list.

print_Satisf_Signals(phi); Displays the satisfaction signal of a given formula.
*********************
This is for Multilayer perceptron (MLP)
- Weights and file should be store in the same format as given:
- weight for each line are ponderation of the output of each (layer-1) neurons and similarly for biases proprietary to one neuron in each layer.
- Activation function for hidden layer can be changed in DnnAff.cpp by commenting or un-commenting each corresponding one, *activation function should be the same in all hidden layers*.
- Output set is computed from:
std::vector<Affine2Vector> output = DeepNeuralNetwork_aff(Inputs , ".../weights.txt", ".../biases.txt");
Inputs should be a vector of Affine form (std::vector<Affine2Vector>) each associated to one input of the NN.
****************
Image: Reachable tube of a nonlinear car model driven by a Neural Network in closed loop using Dynibex
<p align="center">
  <img src="NonlinearcarNNCS.png" alt="Nonlinear car NNCS" width="70%">
</p>

