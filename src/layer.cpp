#include <layer.hpp>

namespace MCIA
{
    Layer::Layer(int connections_in_count,
                 int connections_out_count,
                 double (*activation_function)(double),
                 double (*d_activation_function)(double),
                 double alpha,
                 int m)
        : mfp_activation_function(std::ref(activation_function)),
          mfp_d_activation_function(std::ref(d_activation_function))
    {
        m_alpha = alpha;
        m_m = m;
        m_ones = Eigen::MatrixXd(1, m);
        m_ones_transpose = Eigen::MatrixXd(m, 1);
        m_bias = Eigen::MatrixXd::Zero(connections_out_count, 1);
        m_weight = Eigen::MatrixXd::Constant(connections_out_count, connections_in_count, 1);
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, 1 / sqrt(connections_in_count));
        m_weight.unaryExpr(std::ref(distribution));
    }

    Layer::~Layer() {}

    Eigen::MatrixXd Layer::forward(Eigen::MatrixXd input_vector)
    {
        m_last_input = input_vector;
        m_last_z = m_weight * input_vector + m_bias * m_ones;
        return m_last_z.unaryExpr(mfp_activation_function);
    }

    Eigen::MatrixXd Layer::backward(Eigen::MatrixXd delta,
                                    Eigen::MatrixXd weight)
    {
        m_last_delta = weight.transpose() * delta * m_last_z.unaryExpr(mfp_d_activation_function);
        return m_last_delta;
    }

    void Layer::update()
    {
        m_weight -= (m_alpha / m_m) * (m_last_delta * m_last_input.transpose());
        m_bias -= (m_alpha / m_m) * m_last_delta * m_ones_transpose;
    }
}