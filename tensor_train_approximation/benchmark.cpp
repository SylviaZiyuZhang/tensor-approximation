#include <xerus.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace xerus;

TTTensor randomTTSVD(const Tensor& _x,
  const std::vector<size_t>& _ranks, const std::vector<size_t>& _oversampling)
{
  std::normal_distribution<double> dist(0, 1);

  const size_t d = _x.degree();
  TTTensor u(d);
  Tensor b = _x;

  for(size_t j = d; j >= 2; --j) {
      const size_t s = _ranks[j-2] + _oversampling[j-2];

      const std::vector<size_t> mixDims(b.dimensions.cbegin(), b.dimensions.cbegin()+(j-1));

      std::vector<size_t> outDims({s});
      outDims.insert(outDims.end(), b.dimensions.cbegin()+(j-1), b.dimensions.cend());

      Tensor a(outDims, Tensor::Representation::Sparse, Tensor::Initialisation::Zero);

      if(b.is_sparse()) {
          const size_t staySize = misc::product(b.dimensions, j-1, b.dimensions.size());

          std::map<size_t, std::vector<value_t>> usedG;

          const auto& data = b.get_sparse_data();
          for(const auto& entry : data) {
              const size_t pos = entry.first/staySize;
              const size_t outPos = entry.first%staySize;

              auto& gEntry = usedG[pos];
              if(gEntry.empty()) {
                  gEntry.reserve(s);
                  for(size_t k = 0; k < s; ++k) {
                      gEntry.push_back(dist(xerus::misc::randomEngine));
                  }
              }

              for(size_t k = 0; k < s; ++k) {
                  a[outPos+k*staySize] += gEntry[k]*entry.second;
              }
          }

      } else {
          std::vector<size_t> gDims({s});
          gDims.insert(gDims.end(), mixDims.cbegin(), mixDims.cend());
          const Tensor g = Tensor::random(gDims, dist, xerus::misc::randomEngine);
          contract(a, g, false, b, false, j-1);
      }


      Tensor R, Q;
      calculate_rq(R, Q, a, 1);


      if(j == d) {
          contract(b, b, false, Q, true, 1);
          std::vector<size_t> Q_new_dimensions = std::vector<size_t>(Q.dimensions);
          Q_new_dimensions.push_back(1);
          Q.reinterpret_dimensions(Q_new_dimensions);
          u.set_component(j-1, Q);
      } else {
          contract(b, b, false, Q, true, 2);
          u.set_component(j-1, Q);
      }
  }
  std::vector<size_t> b_new_dimensions = std::vector<size_t>({1});
  b_new_dimensions.insert(b_new_dimensions.end(), b.dimensions.begin(), b.dimensions.end());
  b.reinterpret_dimensions(b_new_dimensions);
  u.set_component(0, b);

  u.round(_ranks);

  return u;
}

int main()
{
  Tensor t1 = Tensor::random({100, 15});
  Tensor t2 = Tensor::random({15, 100, 15});
  Tensor t3 = Tensor::random({15, 100});
  Tensor T = contract(contract(t1, false, t2, false, 1), false, t3, false, 1);
  Tensor S = Tensor::random({100, 100, 100});
  std::vector<size_t> r = std::vector<size_t>({10, 10});
  std::vector<size_t> oversampling = std::vector<size_t>({10, 10});

  auto start = std::chrono::high_resolution_clock::now();
  TTTensor res = randomTTSVD(T, r, oversampling);
  auto end = std::chrono::high_resolution_clock::now();
  std::ofstream outStream = std::ofstream("res.csv");
  std::cout << "Hello, world!" << std::endl;
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " microseconds used." << std::endl;
  misc::stream_writer(outStream, res, misc::FileFormat::TSV);

  Tensor T2(res);

  TTTensor quasi_T(T);
  std::cout << "Approximation rank:";
  for (auto i : res.ranks()) {
    std::cout << i << ",";
  }
  std::cout << std::endl;
  std::cout << "Approximation error: " << frob_norm(T2 - T) / frob_norm(T) <<std::endl;
  std::cout << "Best apx rank:";
  for (auto i : quasi_T.ranks()) {
    std::cout << i << ",";
  }
  std::cout << std::endl;
  std::cout << "Best apx error:" << frob_norm(Tensor(quasi_T) - T) / frob_norm(T) <<std::endl;
  std::cout << "Random error:" << frob_norm(S - T) / frob_norm(T) <<std::endl;
  return 0;
}
