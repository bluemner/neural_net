#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <cassert>
#include <cmath>
namespace betacore
{
	
	
	struct Connection{
		double weight;
		double delta_weight;
	};

	template <typename T>
	class Neuron;
	typedef std::vector<Neuron<double>> Layer;


	template <typename T>
	class Neuron {
		private:
			const T eta = (T) 0.15; // [0.0...1.0] overall net training rate
			const T alpha = (T) 0.5;// [0.0...n] multiplier of last weight change (momentum)
			unsigned  m_index;
			T m_output_value;
			T m_gradient;
		
			std::vector<Connection> m_output_weights;
			T random_weight(){				// For help see:: http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution

				std::random_device rd;  //Will be used to obtain a seed for the random number engine
				std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
				std::uniform_real_distribution<> dis(0, 0); // number between 0,1
				return (T) dis(gen);//Each call to dis(gen) generates a new random double
			}
			static T transformation_function(T x){
				// tanh - output range [ -1.0 .... 1.0]
				// Note: make sure to scale out put into range
				return tanh(x);
			}
			static T transformation_function_derivative(T x){
				// approximation of tanh : 1 - x^2
				// the real d/dx tanh x  = 1 - tanh^2 x
				return 1.0 - x * x;
			}
			
		public:
			Neuron(unsigned number_of_outputs, unsigned index){
				for(unsigned i =0; i<number_of_outputs; i++){

					m_output_weights.push_back(Connection());
					m_output_weights.back().weight = random_weight();

				}
				this-> m_index= index;
			}
			void set_output_value(T value){
				this->m_output_value = value;
			}
			T get_output_value() const{
				return this->m_output_value;
			}
			void feed_forword(const Layer &previous_layer){
				double sum = 0.0;
				for(unsigned i =0; i< previous_layer.size(); ++i){
					sum += previous_layer[i].get_output_value() * previous_layer[i].m_output_weights[m_index].weight;
				}
				m_output_value = Neuron::transformation_function(sum);
			}
			void calculate_output_gradients(T target_value){
				// keeps traning on the path of reducing error
				T delta  = target_value - m_output_value;
				m_gradient = delta * Neuron::transformation_function_derivative(m_output_value);
			}
			void calculate_hidden_gradients(const Layer &next_layer){
				T down =  sum_down_stream(next_layer);
				m_gradient = down * Neuron::transformation_function_derivative(m_output_value);
			}
			T sum_down_stream(const Layer &next_layer) const{
				T sum =0.0;

				for(unsigned i =0; i< next_layer.size()-1 ; ++i){
					sum += m_output_weights[i].weight * next_layer[i].m_gradient;
				}
				return sum;
			}
			void update_input_weights(Layer &previous_layer){
				// The weights to be updated are in the pervious layer
				for(unsigned i=0; i < previous_layer.size(); ++i){
					Neuron &neuron = previous_layer[i];
					T old_delta_weight = neuron.m_output_weights[m_index].delta_weight;
					T new_delta_weight = 
						// Individual input, magnified by the gradient and training rate;
						// eta - over all training rate
						eta  
						* neuron.get_output_value() 
						* m_gradient
						// Also add momentum = a fraction of the previous delta weight
						* alpha
						* old_delta_weight;
					neuron.m_output_weights[m_index].delta_weight = new_delta_weight;
					neuron.m_output_weights[m_index].weight += new_delta_weight;
				} 
			}
	};

	template <typename T>
	class Net{
		private:
			std::vector<Layer> m_layers; 
			T m_error=0.0;
			T m_recent_average_error;
			T m_recent_average_smoothing_factor;
		public:
			Net(const std::vector<unsigned> &topology){
				unsigned number_of_layers = topology.size();
				for(unsigned  i = 0; i = topology.size(); i++){

					m_layers.push_back(Layer());
					unsigned number_of_outputs = number_of_layers == topology.size() -1 ? 0 : topology[i+1];
					// <= because of 1 bias Neuron
					for(unsigned j =0; j<= topology[i]; j++){
						//get last element
						m_layers.back().push_back(Neuron<T>(number_of_outputs, j));
						std::cout<< "Made a neuron" << std::endl;
 					}
					 // force the basis be 1.0
					 m_layers.back().back().set_output_value(1.0);
				}
			}
			void feed_forword(const std::vector<T> &input_values){
				assert(input_values.size()== m_layers[0].size()-1);// -1 for bias neuron 
				for(unsigned i =0;  i< input_values.size(); ++i){
					m_layers[0][1].set_output_value(input_values[i]);
				}
				// forward propagation
				for(unsigned i =0;  i< m_layers.size(); ++i){
					Layer &previous_layer = m_layers[i-1]; // previous layer
					for(unsigned j =0; j < m_layers [i].size() -1 ; ++j){
						m_layers[i][j].feed_forword(previous_layer);

					}
				}
			}
			void back_prop(const std::vector<T> &target_values){
				// Calculate overall net error (RMS of output net error) [RMS = "Root Mean Square Error"]
				betacore::Layer &output_layer = m_layers.back();
				this->m_error = 0.0;

				for(unsigned i =0; i < output_layer.size(); i++){
					T delta = target_values[i] - output_layer[i].get_output_value();
					m_error += delta * delta;
				}
				this->m_error /= output_layer.size()-1; // get average error squared
				this->m_error = std::sqrt(m_error); // RMS

				// Implement a recent average measurement
				m_recent_average_error  = (m_recent_average_error * m_recent_average_smoothing_factor + m_error) / (m_recent_average_smoothing_factor + 1.0);

				// Calculate output layer gradients 
				for(unsigned i =0; i< output_layer.size() -1; ++i){
					output_layer[i].calculate_output_gradients(target_values[i]);
				}
				// Calculate gradients on hidden layer
				for(unsigned i = m_layers.size() -2; i>0; i--){
					Layer &hidden_layer = m_layers[i];
					Layer &next_layer = m_layers[i+1];
					for(unsigned j =0; j< hidden_layer.size(); j++){
							hidden_layer[i].calculate_hidden_gradients(next_layer);
					}
				}
				// For all layers from outputs to first hidden layer 
				// update connection weights
				for( unsigned i = m_layers.size()-1; i > 0 ; --i ){
					Layer &layer = m_layers[i]; 
					Layer &previous_layer = m_layers[i-1];
					for(unsigned j=0; j< layer.size()-1; ++j){
						layer[j].update_input_weights(previous_layer);
					}
				}
			}
			void get_results(std::vector<T> &result_values) const{
					result_values.clear();
					for(unsigned n=0; n<m_layers.back().size()-1;++n){
						result_values.push_back(m_layers.back()[n].get_output_value());
					}
			}
	};
}

// Main method
int main(int argc, char* argv[])
{
	//eg... { 3,2,1 } 
	std::vector<unsigned> topology = {3,2,1};	
	betacore::Net<double> myNet(topology);

	// input values :: need to hold a bunch of doubles.
	std::vector<double> intput_values;
	myNet.feed_forword(intput_values);

	std::vector<double> target_values;
	myNet.back_prop(target_values);

	std::vector<double> result_values;
	myNet.get_results(result_values);

}