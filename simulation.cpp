#include "ibex.h"
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <chrono>
#include <string>
#include "DnnAff.h"

using namespace ibex;
using namespace std;

#define __PREC__ 1e-7
#define __METH__ HEUN
#define __DURATION__ 0.1
#define NUM_MOVES 8
#define PATH_LEN 0.4323323584
#define X_ORIG 0.0
#define Y_ORIG 0.0
   // Define movement vectors
    const int movements[NUM_MOVES][3] = {
        {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {-1, 1, 0},
        {-1, 0, 0}, {-1, -1, 0}, {0, -1, 0}, {1, -1, 0}
    };

std::vector<std::vector<double>> read_file(const std::string& file_path) {
    std::vector<std::vector<double>> data;
    std::ifstream file(file_path);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (ss >> value) {
            row.push_back(std::stod(value));
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }
    return data;
}

std::vector<std::pair<IntervalVector, Interval>> extract_jn_box(ibex::simulation& sim) {
    std::vector<std::pair<IntervalVector, Interval>> jn_box;
    for (const auto& sol : sim.list_solution_g) {
        if (sol.box_jn) { // Ensure it's not NULL
            jn_box.push_back({*sol.box_j1,sol.time_j}); // Dereference pointer
        }
    }

    return jn_box;
}

Affine2Vector mode_dyn(const int& num_move, const Affine2Vector& In_affx, const Affine2Vector& In_affy, const double& X_orig, const double& Y_orig)
{
  IntervalVector yinit(4);
  Variable y(4);
  const int Ka = 2;
  const double Kpref = 1;
  double Xway, Yway;
  yinit[0] = Interval(0,0); // erreur position
  yinit[1] = Interval(0,0);
  yinit[2] = Interval(X_orig); ///modifier pour revenir à 0 pour l'origine
  yinit[3] = Interval(Y_orig);

  AF_fAFFullI::setAffineNoiseNumber(60);
  Affine2Vector yinit_aff(yinit, true);

  yinit_aff[0] = In_affx[0];
  yinit_aff[1] = In_affy[0];

  Xway = X_orig + 0.5*movements[num_move-1][0];
  Yway = Y_orig + 0.5*movements[num_move-1][1];
  // modèle dynamique en interval, Ka*(Xref - X)
  Interval Ax = Interval(1) + Interval(-0.00003, 0.00003); // coeff commande en vitesse
  Interval Bx = Interval(-0.0002, 0.0002);
    
    Function ydot = Function(y, Return(
    Ax*(Ka*(-y[0]) + Kpref*(Xway) + y[2]*(Ka-Kpref)) + Bx,
    Ax*(Ka*(-y[1]) + Kpref*(Yway) + y[3]*(Ka-Kpref)) + Bx,
    Kpref*(Xway - y[2]),
    Kpref*(Yway - y[3])
  ));

  ivp_ode problem = ivp_ode(ydot, 0, yinit_aff, SYMBOLIC);
  simulation simu = simulation(&problem, 2, HEUN, 1e-3, 0.01);
  
  simu.run_simulation();
  yinit_aff = simu.get_last_aff();

  return yinit_aff;
}


std::pair<Affine2Vector, std::vector<std::pair<IntervalVector, Interval>>> mode_dyn_2nd_order(const int& num_move, const Affine2Vector& In_affx, const Affine2Vector& In_affy, const Affine2Vector& SPD_xy , const double& X_orig, const double& Y_orig)
{
  IntervalVector yinit(6);
  Variable y(6);
  const int Ka = 2;
  const double Kpref = 1;
  double Xway, Yway;
  yinit[0] = Interval(0,0); // erreur position
  yinit[1] = Interval(0,0);
  yinit[2] = Interval(0,0); // erreur position
  yinit[3] = Interval(0,0);
  yinit[4] = Interval(X_orig,X_orig); ///modifier pour revenir à 0 pour l'origine
  yinit[5] = Interval(Y_orig,Y_orig);

  AF_fAFFullI::setAffineNoiseNumber(100);
  Affine2Vector yinit_aff(yinit, true);

  yinit_aff[0] = In_affx[0];
  yinit_aff[1] = In_affy[0];
  yinit_aff[2] = SPD_xy[0];
  yinit_aff[3] = SPD_xy[1];

  std::cout<< yinit_aff<<std::endl;

  Xway = X_orig + 0.5*movements[num_move-1][0];
  Yway = Y_orig + 0.5*movements[num_move-1][1];
  // modèle dynamique en interval, Ka*(Xref - X)
  //Interval Ax = Interval(16) + Interval(0, 0); // coeff commande en vitesse
  const double Ax = 16;
  Interval Bx = Interval(-1.9, 1.9);
    
    Function ydot = Function(y, Return(
    y[2], y[3],
    Ax*(Ka*(-y[0]) + Kpref*(Xway) + y[4]*(Ka - Kpref) - y[2]) + Bx,
    Ax*(Ka*(-y[1]) + Kpref*(Yway) + y[5]*(Ka - Kpref) - y[3]) + Bx,
    Kpref*(Xway - y[4]),
    Kpref*(Yway - y[5])
  ));

  ivp_ode problem = ivp_ode(ydot, 0, yinit_aff, SYMBOLIC);
  simulation simu = simulation(&problem, 2, HEUN, 1e-5, 0.01);
  
  simu.run_simulation();
  yinit_aff = simu.get_last_aff();

  std::vector<std::pair<IntervalVector, Interval>> jn_box = extract_jn_box(simu);

// Print the extracted IntervalVectors
//for (const auto& iv : jn_box) {
//    std::cout << iv << std::endl;
//}

  return {yinit_aff,jn_box};
}

std::vector<int> mode_classifier(const std::vector<Affine2Vector>& out_dnn)
{
  std::vector<int> mode_list;
  int max_mode=0;
  double lb_max_mode=-1000;
  double lb_min_mode;
  while (max_mode ==0) //on connait pas le nombre mini de l'output du NN
  {    
    lb_max_mode = lb_max_mode*10;
  for (size_t i = 0; i < out_dnn.size(); i++)
  {
    if (out_dnn[i][0].itv().ub() > lb_max_mode)
    {
      max_mode=i+1;
      lb_max_mode = out_dnn[i][0].itv().ub(); //upper bound of interval projection of affine output
      lb_min_mode = out_dnn[i][0].itv().lb(); //lower bound of interval projection of affine output
    }
  }
  }

  mode_list.push_back(max_mode);

  for (size_t i = 0; i < out_dnn.size(); i++)
  {
    if (out_dnn[i][0].itv().ub() > lb_min_mode && i+1 != max_mode)
    {
      mode_list.push_back(i+1);
    }
  }
  
  return mode_list;
}

/*
std::vector<int> output_U (const std::vector<int>& input, const std::vector<int>& concatenated)
{
    if (concatenated.empty()){
return input;
  }
  std::vector<int> mode_list = concatenated;
  for (size_t i = 0; i < concatenated.size(); i++)
  {
    for (size_t j = 0; j < input.size(); j++)
    {
      if (input[j]!=concatenated[i])
      {
        mode_list.push_back(input[j]);
      }
    }
  }
//fait en sorte que le vecteur ne s'append pas trop!!  
  return mode_list;
}
*/
std::vector<int> output_U(const std::vector<int>& input, const std::vector<int>& concatenated) {
    std::vector<int> mode_list = concatenated; // Copy the concatenated elements

    for (int value : input) {
        // Check if value is already in mode_list
        bool exists = false;
        for (int existing_value : mode_list) {
            if (existing_value == value) {
                exists = true;
                break;
            }
        }
        // Add only if it's not already in the list
        if (!exists) {
            mode_list.push_back(value);
        }
    }

    return mode_list;
}



 std::vector<std::pair<double, double>> origin_builder(const std::vector<int>& in_mode, const double& X_orig, const double& Y_orig){
  
  std::vector<std::pair<double, double>> result;
  for (size_t i = 0; i < in_mode.size(); i++)
  {//c'est la ou ça part en zbeul
    result.push_back({X_orig + PATH_LEN*movements[in_mode[i]-1][0],Y_orig + PATH_LEN*movements[in_mode[i]-1][1]});
  }
  return result;

 }

 std::vector<int> DNN_Path_finder(const std::vector<Affine2Vector>& inputs)
 {
    const std::string weightsFile = "/home/antoine-bc/Desktop/AffineNeuralNetwork/weights.txt";
    const std::string biasesFile = "/home/antoine-bc/Desktop/AffineNeuralNetwork/biases.txt";
    std::vector<int> mode_list = {};
    //Affine2Splitter(transformed_input);
    const int n_pavage = 2;
      //on découpe l'espace de départ pour raffiner la prédiction
      /*
      for (size_t i = 0; i < inputs.size(); i++)
      {
        std::cout<<"--in "<<i<<": "<<inputs[i]<<std::endl;
      }
      */

    std::vector<std::vector<Affine2Vector>> paved_inputs = Affine2Splitter(inputs ,n_pavage);
    std::vector<Affine2Vector> Dnn_output;
    for (size_t i = 0; i < paved_inputs.size(); i++)
    {
          std::cout<<" | "<< paved_inputs[i][0] << " |pavage| " << paved_inputs[i][1];
          std::cout<< std::endl;

          Dnn_output = DeepNeuralNetwork_aff(paved_inputs[i], weightsFile, biasesFile);
          mode_list = output_U(mode_classifier(Dnn_output),mode_list);
          //std::cout << "Final reachable states:" << std::endl;
          //for (size_t f = 0; f < Dnn_output.size(); ++f) {
          //std::cout << "Reachable state for neuron " << f+1 << " : " << Dnn_output[f] << std::endl;
          //}  
          for (size_t j = 0; j < mode_list.size() ; j++)
          {
          std::cout<< "predicted mode: " << mode_list[j] << std::endl;
          }
    }
          
          //for (size_t j = 0; j < mode_list.size() ; j++)
          //{
          //std::cout<< "predicted mode: " << mode_list[j] << std::endl;
          //}
  return mode_list;
 }

 std::vector<int> DNN_Path_finder_bisec(const std::vector<Affine2Vector>& inputs)
 {
    const std::string weightsFile = "/home/antoine-bc/Desktop/AffineNeuralNetwork/weights.txt";
    const std::string biasesFile = "/home/antoine-bc/Desktop/AffineNeuralNetwork/biases.txt";
    std::vector<int> mode_list = {};
    std::vector<int> mode_list_temp = {};
    const int n_pavage = 3;
      //on découpe l'espace de départ pour raffiner la prédiction
      /*
      for (size_t i = 0; i < inputs.size(); i++)
      {
        std::cout<<"--in "<<i<<": "<<inputs[i]<<std::endl;
      }
      */

    std::vector<std::vector<Affine2Vector>> paved_inputs = Affine2Splitter(inputs, n_pavage);
    std::vector<std::vector<Affine2Vector>> paved_inputs_temp;
    std::vector<Affine2Vector> Dnn_output;
    for (size_t i = 0; i < paved_inputs.size(); i++)
    {
          mode_list_temp = {};
          std::cout<<" | "<< paved_inputs[i][0] << " |pavage| " << paved_inputs[i][1];
          std::cout<< std::endl;

          Dnn_output = DeepNeuralNetwork_aff(paved_inputs[i], weightsFile, biasesFile);
          mode_list_temp = mode_classifier(Dnn_output);
          std::cout<<mode_list_temp.size()<<std::endl;

          if (mode_list_temp.size()>1)
          {
            paved_inputs_temp = Affine2Splitter(paved_inputs[i], n_pavage);
            for (size_t j = 0; j < paved_inputs_temp.size(); j++)
            {
              std::cout<<" | "<< paved_inputs_temp[i][0] << " |sous-pavage| " << paved_inputs_temp[i][1];
              std::cout<< std::endl;
              Dnn_output = DeepNeuralNetwork_aff(paved_inputs_temp[i], weightsFile, biasesFile);
              mode_list = output_U(mode_classifier(Dnn_output),mode_list); //union
            }
          }
          else
          {
            mode_list = output_U(mode_list_temp,mode_list); //union
          }
          
    }
          
          for (size_t j = 0; j < mode_list.size() ; j++)
          {
          std::cout<< "predicted mode: " << mode_list[j] << std::endl;
          }
  return mode_list;
 }

struct node {
  int branche_index;
  std::vector<Affine2Vector> state;
  std::vector<int> modes;
  std::pair<double, double> MP_pos;
  std::vector<Affine2Vector> Spdxyaff;
  std::vector<int> branche_liste; //idée pour retracer dans l'arbre le tubes avec les dépendances
  std::vector<std::pair<IntervalVector, Interval>> Jn_vector;

//rajouter un attribut avec l'index dans un split de la sous boite correspondant à une partie du set initiale
      node(const int& branche, const std::vector<int>& branch_id, const std::vector<Affine2Vector>& inputs,const int& n_mode, const std::pair<double, double>& in_MPxy, const std::vector<Affine2Vector>& Spdxy_aff){
      //le constructeur à pour but de construire un nouveau noeuds à partir de celui d'avant et du mode choisi
      branche_index = branche;
      std::cout<< " mode de simu: " << n_mode<< endl;
         //std::cout<< time_index << inputs[0]<< n_mode << in_MPxy.first<< Spdxy_aff[0][0] <<std::endl;
      std::pair<Affine2Vector, std::vector<std::pair<IntervalVector, Interval>>> Outsim = mode_dyn_2nd_order(n_mode, inputs[0], inputs[1], Spdxy_aff[0] ,in_MPxy.first, in_MPxy.second);
//y'a pb je crois qu'on prend un num de mode 1 en retard
      Affine2Vector Outdyn = Outsim.first;
      Jn_vector = Outsim.second;
      IntervalVector Spd_xy(2);
      Spd_xy[0] = Interval(0);
      Spd_xy[1] = Interval(0);
      Affine2Vector Spd_xy_aff(Spd_xy, true);  

      //on met à jour les entrée du réseaux de neurones
      state = inputs;
      state[0][0] = Outdyn[0];
      state[1][0] = Outdyn[1];

      //on passe par un interval2vecteur intermédiaire pour créer l'objet Spdxy
        Spd_xy_aff[0] = Outdyn[2];
        Spd_xy_aff[1] = Outdyn[3];
        Spdxyaff = {Spd_xy_aff};
        modes = DNN_Path_finder_bisec(state);
      MP_pos = {in_MPxy.first + PATH_LEN*movements[n_mode-1][0],in_MPxy.second + PATH_LEN*movements[n_mode-1][1]};
      branche_liste = branch_id;
      branche_liste.push_back(branche); //on stocke les index pour chaque pas de temps d'avancement dans le tableau de l'arbre
      }
};
void printFinalBranching(const std::vector<std::vector<std::pair<IntervalVector, Interval>>>& Final_branching) {
    std::cout << "\n=== Final Branching Output ===\n";

    for (size_t i = 0; i < Final_branching.size(); i++) { // Fixed: Use .size() instead of .first.size()
        std::cout << "Branch " << i + 1 << " (Positions to reach node):\n";

        if (Final_branching[i].empty()) { // Fixed: Use Final_branching[i].empty()
            std::cout << "  [Empty Path]\n";
        }

        for (size_t j = 0; j < Final_branching[i].size(); j++) {
            // Fixed: Use Final_branching[i][j].first (IntervalVector) and Final_branching[i][j].second (Interval)
            std::cout << "  Step " << j + 1 << ": " << Final_branching[i][j].first 
                      << " , time: " << Final_branching[i][j].second << std::endl;
        }

        std::cout << "--------------------------------\n";
    }
}

void recordFinalBranching(const std::vector<std::vector<std::pair<IntervalVector, Interval>>>& Final_branching) {
    std::cout << "\n=== Final Branching Output ===\n";
          std::ofstream branch_result;
    branch_result.open("branch_result.txt");
    for (size_t i = 0; i < Final_branching.size(); i++) { // Fixed: Use .size() instead of .first.size()
        std::cout << "Branch " << i + 1 << " (Positions to reach node):\n";

        if (Final_branching[i].empty()) { // Fixed: Use Final_branching[i].empty()
            std::cout << "  [Empty Path]\n";
        }

        for (size_t j = 0; j < Final_branching[i].size(); j++) {
            // Fixed: Use Final_branching[i][j].first (IntervalVector) and Final_branching[i][j].second (Interval)
            std::cout << "  Step " << j + 1 << ": " << Final_branching[i][j].first 
                      << " , time: " << Final_branching[i][j].second << std::endl;
                    if (Final_branching[i][j].first[0].lb()>1e-4 && Final_branching[i][j].first[1].lb()>1e-4 && j%50==0)
                    {
                      branch_result << Final_branching[i][j].first[0] <<" ; "<<Final_branching[i][j].first[1] <<std::endl; //branching result là
                    }
                                        
        }

        std::cout << "--------------------------------\n";
    }
    branch_result.close();
}

//function pour enlever les intervalles de temps nulles
std::vector<std::vector<std::pair<IntervalVector, Interval>>> remove_null_intervals(
    const std::vector<std::vector<std::pair<IntervalVector, Interval>>>& Real_time_branching) {
    
    std::vector<std::vector<std::pair<IntervalVector, Interval>>> Filtered_branching;

    for (const auto& branch : Real_time_branching) {
        std::vector<std::pair<IntervalVector, Interval>> filtered_branch;
        
        for (const auto& segment : branch) {
            const Interval& time_interval = segment.second;
            
            // Check if interval is null
            if (time_interval.lb() != time_interval.ub()) {
                filtered_branch.push_back(segment); // Keep only valid intervals
            }
        }
        
        // Add non-empty branches to the final list
        if (!filtered_branch.empty()) {
            Filtered_branching.push_back(filtered_branch);
        }
    }

    return Filtered_branching;
}

// Function to reconstruct real time
std::vector<std::vector<std::pair<IntervalVector, Interval>>> reconstruct_real_time(
    const std::vector<std::vector<std::pair<IntervalVector, Interval>>>& Final_branching) {
    
    std::vector<std::vector<std::pair<IntervalVector, Interval>>> Real_time_branching = Final_branching;

    for (auto& branch : Real_time_branching) {
        double cumulative_time = 0.0; // Initialize real time counter for each branch
        for (auto& segment : branch) {
            Interval& time_interval = segment.second;

            // Compute the duration using the framework's methods
            double duration = time_interval.ub() - time_interval.lb();
            if (time_interval.lb()<0)
            {
              duration = 0; //les boxs jn sont initialisé entre -0.01 et 0 ce qui crée un décalage dans toutes la branche
            }
            
            // Update time interval with accumulated real time
            double new_lb = cumulative_time;
            double new_ub = cumulative_time + duration;
            if (new_lb<0)
            {
              time_interval= Interval(0, new_ub); //un temps négatif n'a aucun sens
            }
            else
            {
              time_interval= Interval(new_lb, new_ub);
            }

            // Accumulate time for the next segment
            cumulative_time = new_ub;
        }
    }
    Real_time_branching = remove_null_intervals(Real_time_branching);
    return Real_time_branching;
}

int main(){
    //  std::ofstream branch_result;
   // branch_result.open("branch_result.txt");

  const int order = 2;
  std::vector<int> mode_list;
  std::vector<std::pair<double, double>> origin_list;
  Affine2Vector Outdyn(0,true);
  double Xway, Yway;
  double X_orig = X_ORIG;
  double Y_orig = Y_ORIG;
/*
  IntervalVector tempPosx(1);
  tempPosx[0] = Interval(0,0);
  IntervalVector tempPosy(1);  
  tempPosy[0] = Interval(0,0); 
    Affine2Vector tempPosx_aff(tempPosx, true);
    Affine2Vector tempPosy_aff(tempPosy, true);
*/
  int num_move=2;

  if (order==1)
  {
  IntervalVector yinit(4);
  yinit[0] = Interval(0,0);
  yinit[1] = Interval(0,0);
  yinit[2] = Interval(0,0);
  yinit[3] = Interval(0,0);
    Affine2Vector Outdyn(yinit, true);
  }
  else
  {
  IntervalVector yinit(6);
  yinit[0] = Interval(0,0);
  yinit[1] = Interval(0,0);
  yinit[2] = Interval(0,0);
  yinit[3] = Interval(0,0);
  yinit[4] = Interval(0,0);
  yinit[5] = Interval(0,0);
    Affine2Vector Outdyn(yinit, true);
  }



//std::cout<<"je suis la"<<std::endl;  
IntervalVector Spd_xy(2);
Spd_xy[0] = Interval(-0.01, 0.01);
Spd_xy[1] = Interval(-0.01, 0.01);
Affine2Vector Spd_xy_aff(Spd_xy, true);  


IntervalVector In0(1);
In0[0] = Interval(X_orig, X_orig+0.01);
Affine2Vector In_0(In0, true);  

IntervalVector In1(1);
In1[0] = Interval(Y_orig, Y_orig+0.01);
Affine2Vector In_1(In1, true);

IntervalVector In2(1);
In2[0] = Interval(3.5, 3.5);
Affine2Vector In_2(In2, true);

IntervalVector In3(1);
In3[0] = Interval(3.5, 3.5);
Affine2Vector In_3(In3, true);

IntervalVector In4(1);
In4[0] = Interval(1.5, 1.5);
Affine2Vector In_4(In4, true);

IntervalVector In5(1);
In5[0] = Interval(1.65, 1.65);
Affine2Vector In_5(In5, true);

IntervalVector In6(1);
In6[0] = Interval(2.5, 2.5);
Affine2Vector In_6(In6, true);

IntervalVector In7(1);
In7[0] = Interval(3, 3);
Affine2Vector In_7(In7, true);

   std::vector<Affine2Vector> inputs = {In_0,In_1,In_2,In_3,In_4,In_5,In_6,In_7}; //input layer

    std::vector<std::vector<node>> final_tree;  // Tree of nodes
    std::vector<node> temp_row;  // Temporary row of nodes
    std::cout<<"-----------------num noeud : "<<1<<" , pas de temps : "<< 1<<" -----------------------" <<std::endl;
    // Initial node creation
    node In_temp(0, {} , inputs, 2, {X_ORIG, Y_ORIG}, {Spd_xy_aff});
    temp_row.push_back(In_temp);  // First node in row
    final_tree.push_back(temp_row); 
    int horizon = 9;
    int decalage = 0;
    int tmp_dcl = 0;
    // Recursive construction of the row
    for (size_t i = 1; i < horizon; i++) {  
        temp_row = {}; //erasing precedent row
        decalage = 0; //on réinitialise le décalage pour la ranger d'après
        std::cout<<"-----------------------------------------------nouveau pas de temps "<<i+1<<" -----------------------------------------------"<<std::endl;
        for (size_t j = 0; j < final_tree[i-1].size(); j++)
        {
          std::cout<<"nombre de mode noeud precedent : "<<final_tree[i-1][j].modes.size()<<" , nombre noeuds precedent temps : "<< final_tree[i-1].size() << std::endl;
                      //cout<<temp_row[i - 1].MP_pos.first<<"MPposxy origine"<<temp_row[i - 1].MP_pos.second<<endl;

          for (size_t f = 0; f < final_tree[i-1][j].modes.size(); f++)
          {
          std::cout<<"-----------------num noeud : "<<j + f + decalage+1<<" , pas de temps : "<< i + 1<<" -----------------------" << std::endl;
            temp_row.push_back(
            node(
                j + f + decalage, 
                final_tree[i - 1][j].branche_liste,
                final_tree[i - 1][j].state,  // Pass previous node's state
                final_tree[i - 1][j].modes[f], 
                final_tree[i - 1][j].MP_pos, 
                final_tree[i - 1][j].Spdxyaff  // Pass Spdxyaff vector
            ));
          tmp_dcl = f; //on enregistre le nombre de nouveau noeuds crée
          }          
          decalage = decalage + tmp_dcl; // on décale les index des branches avec les nouveaux noeuds

          //rajouter les dépendances entre noeuds utiliser les numéros de branches pour suivre
          //enregistrer les tubes successivements dans un fichier textes pour le diviser en n branches et completer les tubes en dessous.
          ///problème génération des MPs, vu fait normalement
          //peut être utiliser des methodes avec les intervalles pour vérifier la formule stl
          //vérif en fin de simulation on prend pour chaque noeuds la succession d'index qui l'amène à lui puis on prend pour chaque noeuds parent le mouvement qui à permis sa construction

        }   
            final_tree.push_back(temp_row); 
    }
    
   std::vector<std::vector<std::pair<IntervalVector, Interval>>> Final_branching;

const int size_tree = final_tree.size() - 1; // Last index
std::cout << "Size of tree: " << size_tree << std::endl;

// Resize Final_branching to match the number of nodes in the last layer, necessary for insert
Final_branching.resize(final_tree[size_tree].size());

for (size_t i = 0; i < final_tree[size_tree].size(); i++) { 
    std::cout << "Index parent node branch " << i + 1 << " : ";

    for (size_t j = 0; j < final_tree[size_tree][i].branche_liste.size(); j++) {
        int parent_index = final_tree[size_tree][i].branche_liste[j];  // Get parent index to get Jn boxes

        std::cout << parent_index << " > ";

        // Ensure the parent node index is valid
        if (parent_index < final_tree[j].size()) {
            // Append Jn_vector contents to Final_branching[i]
            Final_branching[i].insert(Final_branching[i].end(),
                                      final_tree[j][parent_index].Jn_vector.begin(),
                                      final_tree[j][parent_index].Jn_vector.end()); //maintenant faut récuperer tout les temps, fait
        } else {
            std::cerr << "Warning: Invalid parent index " << parent_index << " for node " << j << std::endl; //ça devrait pas arriver mais au cas où
        }
    }
    std::cout << std::endl;
}

std::vector<std::vector<std::pair<IntervalVector, Interval>>> Real_time_branching = reconstruct_real_time(Final_branching);
//printFinalBranching(Final_branching);
//printFinalBranching(Real_time_branching);
recordFinalBranching(Real_time_branching);
   //  branch_result.close();
    return 0;
}
