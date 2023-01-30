#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <random>

namespace MCIA
{

    class Layer
    {
    public:
        Layer(int connections_in_count,
              int connections_out_count,
              double (*activation_function)(double),
              double (*d_activation_function)(double),
              double alpha,
              int m);
        ~Layer();
        Eigen::MatrixXd forward(Eigen::MatrixXd input_vector);
        Eigen::MatrixXd backward(Eigen::MatrixXd delta, Eigen::MatrixXd weight);
        void update(void);
        Eigen::MatrixXd get_weight(void) { return m_weight; };
        int get_weight_cols(void) { return m_weight.cols(); };
        int get_weight_rows(void) { return m_weight.rows(); };

    private:
        Eigen::MatrixXd m_weight;
        Eigen::VectorXd m_bias;
        Eigen::MatrixXd m_last_delta;
        Eigen::MatrixXd m_last_input;
        Eigen::MatrixXd m_last_z;
        Eigen::MatrixXd m_ones;
        Eigen::MatrixXd m_ones_transpose;
        std::reference_wrapper<double (*)(double)> mfp_activation_function;
        std::reference_wrapper<double (*)(double)> mfp_d_activation_function;
        double m_alpha;
        int m_m;
    };
}

#endif