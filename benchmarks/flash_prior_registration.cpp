// Research-only registration with a pose prior. Not imported by live tracking.
// Uses upstream small_gicp point factors and correspondence search (MIT).
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/rejector.hpp>
#include <small_gicp/util/lie.hpp>
#include <Eigen/Cholesky>

namespace py = pybind11;
namespace sg = small_gicp;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using M6 = Eigen::Matrix<double, 6, 6>;
using V6 = Eigen::Matrix<double, 6, 1>;

sg::PointCloud cloud(const Matrix& xyz, const Matrix& normals, const Matrix& cov) {
  if (xyz.cols()!=3 || normals.rows()!=xyz.rows() || normals.cols()!=3 ||
      cov.rows()!=xyz.rows() || cov.cols()!=9 || xyz.rows()<6 ||
      !xyz.allFinite() || !normals.allFinite() || !cov.allFinite())
    throw std::invalid_argument("finite XYZ, normals and 3x3 covariance rows required");
  sg::PointCloud out;
  out.resize(xyz.rows());
  for (size_t i=0; i<out.size(); ++i) {
    out.point(i) << xyz.row(i).transpose(), 1.;
    out.normal(i) << normals.row(i).transpose(), 0.;
    out.cov(i).setZero();
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) out.cov(i)(r,c)=cov(i,r*3+c);
  }
  return out;
}

// Same units as FLASH's pose filter: camera-axis rotation, and position of
// one fixed object point. Source/target centering is handled by the caller.
V6 difference(const Eigen::Isometry3d& pose, const Eigen::Isometry3d& prior,
              const Eigen::Vector3d& center) {
  Eigen::AngleAxisd rotation(pose.linear()*prior.linear().transpose());
  V6 out;
  out.head<3>() = rotation.axis()*rotation.angle();
  out.tail<3>() = pose*center - prior*center;
  return out;
}

M6 prior_jacobian(const Eigen::Isometry3d& pose, const Eigen::Isometry3d& prior,
                  const Eigen::Vector3d& center) {
  // Fixed 12 tiny pose evaluations, no image search or inner optimizer.
  M6 jac;
  for (int k=0; k<6; ++k) {
    V6 step=V6::Zero(); step[k]=1e-6;
    jac.col(k)=(difference(pose*sg::se3_exp(step),prior,center)-
                difference(pose*sg::se3_exp(-step),prior,center))/(2e-6);
  }
  return jac;
}

struct Prepared {
  sg::PointCloud target, source;
  sg::UnsafeKdTree<sg::PointCloud> tree;
  Prepared(const Matrix& tx, const Matrix& tn, const Matrix& tc,
           const Matrix& sx, const Matrix& sn, const Matrix& sc)
      : target(cloud(tx,tn,tc)), source(cloud(sx,sn,sc)), tree(target) {}

  template<class Factor>
  py::dict solve(const Eigen::Matrix4d& initial, const Eigen::Matrix4d& prediction,
                 const M6& precision, const Eigen::Vector3d& center, double scale,
                 double max_distance, int budget, int threads, double t_eps, double r_eps) {
    if (budget<1 || threads<1 || scale<=0 || !std::isfinite(scale) || max_distance<=0 ||
        !initial.allFinite() || !prediction.allFinite() || !precision.allFinite())
      throw std::invalid_argument("invalid registration input or budget");
    Eigen::Isometry3d pose(initial), prior(prediction);
    sg::ParallelReductionOMP reduction; reduction.num_threads=threads;
    sg::DistanceRejector rejector; rejector.max_dist_sq=max_distance*max_distance;
    std::vector<Factor> factors(source.size());
    int iterations=0, inliers=0;
    bool converged=false;
    double cost=0;
    M6 measurement=M6::Zero();
    {
      py::gil_scoped_release release;
      for (int i=0;i<budget;++i) {
        auto [h,b,e]=reduction.linearize(target,source,tree,rejector,pose,factors);
        inliers=std::count_if(factors.begin(),factors.end(),[](const auto& f){return f.inlier();});
        if(inliers<6) break;
        measurement=scale*h;
        M6 total=measurement;
        V6 gradient=scale*b;
        double prior_cost=0;
        if(precision.cwiseAbs().maxCoeff()>0) {
          const auto residual=difference(pose,prior,center);
          const auto jac=prior_jacobian(pose,prior,center);
          total+=jac.transpose()*precision*jac;
          gradient+=jac.transpose()*precision*residual;
          prior_cost=0.5*residual.dot(precision*residual);
        }
        const V6 step=(total+1e-6*M6::Identity()).ldlt().solve(-gradient);
        if(!step.allFinite()) break;
        pose=pose*sg::se3_exp(step);
        ++iterations;
        cost=scale*e+prior_cost;
        converged=step.head<3>().norm()<r_eps && step.tail<3>().norm()<t_eps;
        if(converged) break;
      }
    }
    py::dict out;
    out["pose"]=pose.matrix(); out["iterations"]=iterations;
    out["inliers"]=inliers; out["converged"]=converged;
    out["last_linearization_information"]=measurement;
    out["last_linearization_cost"]=cost;
    return out;
  }

  py::dict align(const Eigen::Matrix4d& initial, const Eigen::Matrix4d& prediction,
                 const M6& precision, const Eigen::Vector3d& center, double scale,
                 double max_distance, int budget, int threads, double t_eps, double r_eps,
                 const std::string& factor) {
    if(factor=="gicp") return solve<sg::GICPFactor>(initial,prediction,precision,center,scale,max_distance,budget,threads,t_eps,r_eps);
    if(factor=="plane") return solve<sg::PointToPlaneICPFactor>(initial,prediction,precision,center,scale,max_distance,budget,threads,t_eps,r_eps);
    if(factor=="point") return solve<sg::ICPFactor>(initial,prediction,precision,center,scale,max_distance,budget,threads,t_eps,r_eps);
    throw std::invalid_argument("unknown factor");
  }
};

PYBIND11_MODULE(flash_prior_native, m) {
  py::class_<Prepared>(m,"Prepared")
    .def(py::init<const Matrix&,const Matrix&,const Matrix&,const Matrix&,const Matrix&,const Matrix&>())
    .def("align",&Prepared::align);
  m.def("difference",[](const Eigen::Matrix4d& p,const Eigen::Matrix4d& q,const Eigen::Vector3d& c){return difference(Eigen::Isometry3d(p),Eigen::Isometry3d(q),c);});
}
