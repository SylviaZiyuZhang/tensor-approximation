#include <xerus.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace xerus;

Tensor get_countsketch(size_t num_rows, size_t num_cols) {
  Tensor S = Tensor({num_rows, num_cols}, Tensor::Representation::Sparse, Tensor::Initialisation::None);
  thread_local static std::mt19937 eng {std::random_device{}()};
  thread_local static std::uniform_int_distribution<size_t> uni_dist(0, num_rows - 1);
  for (size_t i = 0; i < num_cols; i++) {
    size_t idx = uni_dist(eng);
    S[{idx, i}] = 1.0;
  }
  return S;
}

TTTensor randomTTSVD(const Tensor& _x,
  const std::vector<size_t>& _ranks, const std::vector<size_t>& _oversampling)
{ // oversampling accounts for the bicriteria rank in the paper
  std::normal_distribution<double> dist(0, 1);

  const size_t d = _x.degree();
  TTTensor u(d);
  Tensor b = _x;

  for(size_t j = d; j >= 2; --j) {
      const size_t s = _ranks[j-2] + _oversampling[j-2];
      std::hash<size_t> cntsketch_hash;

      const std::vector<size_t> mixDims(b.dimensions.cbegin(), b.dimensions.cbegin()+(j-1));

      std::vector<size_t> outDims({s});
      outDims.insert(outDims.end(), b.dimensions.cbegin()+(j-1), b.dimensions.cend());

      Tensor a(outDims, Tensor::Representation::Sparse, Tensor::Initialisation::Zero);
      std::vector<size_t> gDims({s});
      gDims.insert(gDims.end(), mixDims.cbegin(), mixDims.cend());
      // Apply CountSketch matricized
      Tensor sk = get_countsketch(s, misc::product(gDims)/s);
      sk.reinterpret_dimensions(gDims);
      contract(a, sk, false, b, false, j-1);

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
