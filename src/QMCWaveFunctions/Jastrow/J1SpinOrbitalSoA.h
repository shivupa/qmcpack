//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                  Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                  Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                  Shiv Upadhyay, shivnupadhyay@gmail.com, University of Pittsburgh
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_ONEBODYSPINJASTROW_OPTIMIZED_SOA_H
#define QMCPLUSPLUS_ONEBODYSPINJASTROW_OPTIMIZED_SOA_H
#include "Configuration.h"
#include "Particle/DistanceTableData.h"
#include "ParticleBase/ParticleAttribOps.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "Utilities/qmc_common.h"
#include "Utilities/IteratorUtility.h"
#include "CPU/SIMD/aligned_allocator.hpp"
#include "CPU/SIMD/algorithm.hpp"
#include <map>
#include <numeric>

namespace qmcplusplus
{
/** @ingroup WaveFunctionComponent
 *  @brief Specialization for one-body Jastrow function using multiple functors
 */
template<class FT>
struct J1SpinOrbitalSoA : public WaveFunctionComponent
{
  ///alias FuncType
  using FuncType = FT;
  ///type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  ///element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  ///use the same container
  using DistRow  = DistanceTableData::DistRow;
  using DisplRow = DistanceTableData::DisplRow;
  ///table index
  const int myTableID;
  ///number of ions
  int Nions;
  ///number of electrons
  const int Nelec;
  /* the number of ion groups if ions in 'Ions' particleset are grouped by species. 0 otherwise.
   * 0 Use slow code path. >= 1 use the code path with ion grouping
   */
  const int NumGroups;
  // number of electron groups
  const int NumTargetGroups;
  ///reference to the sources (ions)
  const ParticleSet& Ions;
  ///reference to the target (elecs)
  const ParticleSet& Elecs;

  ///number of variables this object handles
  int NumVars;
  ///variables handled by this orbital
  opt_variables_type myVars;

  valT curAt;
  valT curLap;
  posT curGrad;

  ///\f$Vat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Vat;
  aligned_vector<valT> U, dU, d2U, d3U;
  aligned_vector<valT> DistCompressed;
  aligned_vector<int> DistIndice;
  Vector<posT> Grad;
  Vector<valT> Lap;
  ///Container for \f$F[ig*NIons+jg]\f$
  std::vector<FT*> J1Functors;
  ///container for the unique Jastrow functions
  std::map<std::string, std::unique_ptr<FT>> J1UniqueFunctors;

  std::vector<std::pair<int, int>> OffSet;
  Vector<RealType> dLogPsi;
  typedef ParticleAttrib<QTFull::GradType> WavefunctionFirstDerivativeType;
  typedef ParticleAttrib<QTFull::ValueType> WavefunctionSecondDerivativeType;
  std::vector<WavefunctionFirstDerivativeType*> gradLogPsi;
  std::vector<WavefunctionSecondDerivativeType*> lapLogPsi;

  J1SpinOrbitalSoA(const std::string& obj_name, const ParticleSet& ions, ParticleSet& els)
      : WaveFunctionComponent("J1SpinOrbitalSoA", obj_name),
        myTableID(els.addTable(ions)),
        Nions(ions.getTotalNum()),
        Nelec(els.getTotalNum()),
        NumGroups(determineNumGroups(ions)),
        NumTargetGroups(determineNumGroups(els)),
        Ions(ions),
        Elecs(els),
        NumVars(0)
  {
    if (myName.empty())
      throw std::runtime_error("J1SpinOrbitalSoA object name cannot be empty!");
    initialize();
  }

  J1SpinOrbitalSoA(const J1SpinOrbitalSoA& rhs) = delete;

  ~J1SpinOrbitalSoA() override
  {
    delete_iter(gradLogPsi.begin(), gradLogPsi.end());
    delete_iter(lapLogPsi.begin(), lapLogPsi.end());
  }

  /* determine NumGroups which controls the use of optimized code path using ion groups or not */
  static int determineNumGroups(const ParticleSet& ions)
  {
    const int num_species = ions.getSpeciesSet().getTotalNum();
    if (num_species == 1)
      return 1;
    else if (num_species > 1 && !ions.IsGrouped)
      return 0;
    else
      return num_species;
  }

  /* initialize storage */
  void initialize()
  {
    J1Functors.resize(Nions * Nelec, nullptr);
    Vat.resize(Nelec);
    Grad.resize(Nelec);
    Lap.resize(Nelec);

    U.resize(Nions);
    dU.resize(Nions);
    d2U.resize(Nions);
    d3U.resize(Nions);
    DistCompressed.resize(Nions);
    DistIndice.resize(Nions);
  }

  void addFunc(int source_type, std::unique_ptr<FT> afunc, int target_type = -1)
  {
    // if target type is not specified J1Functors[i,:] is assigned
    // if target type is specified J1Functors[i,j] is assigned
    //
    // if target type is not specified J1UniqueFunctors["i"] is assigned
    // if target type is specified J1UniqueFunctors["ij"] is assigned
    //
    app_log() << " source" << source_type << " target type " << target_type << std::endl;
    if (target_type == -1)
    {
      for (int i = 0; i < Nions; i++)
        for (int j = 0; j < Nelec; j++)
          if (Ions.GroupID[i] == source_type && J1Functors[i * Nelec + j] == nullptr)
            J1Functors[i * Nelec + j] = afunc.get();
      //if (J1UniqueFunctors[source_type] != nullptr)
      //  delete J1UniqueFunctors[source_type];
      std::stringstream aname;
      aname << source_type;
      J1UniqueFunctors[aname.str()] = std::move(afunc);
    }
    else
    {
      for (int i = 0; i < Nions; i++)
        for (int j = 0; j < Nelec; j++)
          if (Ions.getGroupID(i) == source_type && Elecs.getGroupID(j) == target_type &&
              J1Functors[i * Nelec + j] == nullptr)
            J1Functors[i * Nelec + j] = afunc.get();
      //if (J1UniqueFunctors[source_type] != nullptr)
      //  delete J1UniqueFunctors[source_type];
      std::stringstream aname;
      aname << source_type << target_type;
      J1UniqueFunctors[aname.str()] = std::move(afunc);
    }
  }

  void recompute(const ParticleSet& P) override
  {
    const DistanceTableData& d_ie(P.getDistTable(myTableID));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      computeU3(P, iat, d_ie.getDistRow(iat));
      Vat[iat] = simd::accumulate_n(U.data(), Nions, valT());
      Lap[iat] = accumulateGL(dU.data(), d2U.data(), d_ie.getDisplRow(iat), Grad[iat]);
    }
  }

  LogValueType evaluateLog(const ParticleSet& P,
                           ParticleSet::ParticleGradient_t& G,
                           ParticleSet::ParticleLaplacian_t& L) override
  {
    return evaluateGL(P, G, L, true);
  }

  void evaluateHessian(ParticleSet& P, HessVector_t& grad_grad_psi) override
  {
    const DistanceTableData& d_ie(P.getDistTable(myTableID));
    valT dudr, d2udr2;

    Tensor<valT, DIM> ident;
    grad_grad_psi = 0.0;
    ident.diagonal(1.0);

    for (int iel = 0; iel < Nelec; ++iel)
    {
      const auto& dist  = d_ie.getDistRow(iel);
      const auto& displ = d_ie.getDisplRow(iel);
      for (int iat = 0; iat < Nions; iat++)
      {
        std::stringstream gid;
        gid << Ions.getGroupID(iat) << P.getGroupID(iel);
        if (J1UniqueFunctors.find(gid.str()) == J1UniqueFunctors.end())
        {
          gid.str() = "";
          gid << Ions.getGroupID(iat);
        }
        auto* func = J1UniqueFunctors[gid.str()].get();
        if (func != nullptr)
        {
          RealType r    = dist[iat];
          RealType rinv = 1.0 / r;
          PosType dr    = displ[iat];
          func->evaluate(r, dudr, d2udr2);
          grad_grad_psi[iel] -= rinv * rinv * outerProduct(dr, dr) * (d2udr2 - dudr * rinv) + ident * dudr * rinv;
        }
      }
    }
  }

  PsiValueType ratio(ParticleSet& P, int iat) override
  {
    UpdateMode = ORB_PBYP_RATIO;
    curAt      = computeU(P, iat, P.getDistTable(myTableID).getTempDists());
    return std::exp(static_cast<PsiValueType>(Vat[iat] - curAt));
  }

  inline void evaluateRatios(const VirtualParticleSet& VP, std::vector<ValueType>& ratios) override
  {
    for (int k = 0; k < ratios.size(); ++k)
      ratios[k] = std::exp(Vat[VP.refPtcl] - computeU(VP.refPS, VP.refPtcl, VP.getDistTable(myTableID).getDistRow(k)));
  }

  void evaluateDerivatives(ParticleSet& P,
                           const opt_variables_type& active,
                           std::vector<ValueType>& dlogpsi,
                           std::vector<ValueType>& dhpsioverpsi) override
  {
    evaluateDerivativesWF(P, active, dlogpsi);
    bool recalculate(false);
    std::vector<bool> rcsingles(myVars.size(), false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      int kk = myVars.where(k);
      if (kk < 0)
        continue;
      if (active.recompute(kk))
        recalculate = true;
      rcsingles[k] = true;
    }
    if (recalculate)
    {
      for (int k = 0; k < myVars.size(); ++k)
      {
        int kk = myVars.where(k);
        if (kk < 0)
          continue;
        if (rcsingles[k])
        {
          dhpsioverpsi[kk] = -RealType(0.5) * ValueType(Sum(*lapLogPsi[k])) - ValueType(Dot(P.G, *gradLogPsi[k]));
        }
      }
    }
  }

  void evaluateDerivativesWF(ParticleSet& P, const opt_variables_type& active, std::vector<ValueType>& dlogpsi) override
  {
    bool recalculate(false);
    std::vector<bool> rcsingles(myVars.size(), false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      int kk = myVars.where(k);
      if (kk < 0)
        continue;
      if (active.recompute(kk))
        recalculate = true;
      rcsingles[k] = true;
    }
    if (recalculate)
    {
      const auto& d_table = P.getDistTable(myTableID);
      dLogPsi             = 0.0;
      for (int p = 0; p < NumVars; ++p)
        (*gradLogPsi[p]) = 0.0;
      for (int p = 0; p < NumVars; ++p)
        (*lapLogPsi[p]) = 0.0;
      std::vector<TinyVector<RealType, 3>> derivs(NumVars);

      constexpr RealType cone(1);
      constexpr RealType lapfac(OHMMS_DIM - cone);
      const size_t ns = d_table.sources();
      const size_t nt = P.getTotalNum();

      aligned_vector<int> iadj(nt);
      aligned_vector<RealType> dist(nt);
      std::vector<PosType> displ(nt);

      for (size_t i = 0; i < ns; ++i)
      {
        double cutoff_radius = 0.0;
        for (size_t j = 0; j < nt; ++j)
          if (J1Functors[i * Nelec + j] != nullptr)
            cutoff_radius = std::max(cutoff_radius, J1Functors[i * Nelec + j]->cutoff_radius);
        size_t nn = d_table.get_neighbors(i, cutoff_radius, iadj.data(), dist.data(), displ.data());
        for (size_t nj = 0; nj < nn; ++nj)
        {
          int first(OffSet[i * Nelec + nj].first);
          int last(OffSet[i * Nelec + nj].second);
          bool recalcFunc(false);
          for (int rcs = first; rcs < last; rcs++)
            if (rcsingles[rcs] == true)
              recalcFunc = true;
          if (recalcFunc)
          {
            FT* func = J1Functors[i * Nelec + nj];
            if (func == nullptr)
              continue;
            std::fill(derivs.begin(), derivs.end(), 0);
            if (!func->evaluateDerivatives(dist[nj], derivs))
              continue;
            int j = iadj[nj];
            RealType rinv(cone / dist[nj]);
            PosType& dr = displ[nj];
            for (int p = first, ip = 0; p < last; ++p, ++ip)
            {
              dLogPsi[p] -= derivs[ip][0];
              RealType dudr(rinv * derivs[ip][1]);
              (*gradLogPsi[p])[j] -= dudr * dr;
              (*lapLogPsi[p])[j] -= derivs[ip][2] + lapfac * dudr;
            }
          }
        }
      }
      for (int k = 0; k < myVars.size(); ++k)
      {
        int kk = myVars.where(k);
        if (kk < 0)
          continue;
        if (rcsingles[k])
        {
          dlogpsi[kk] = ValueType(dLogPsi[k]);
        }
      }
    }
  }


  inline valT computeU(const ParticleSet& P, int iat, const DistRow& dist)
  {
    valT curVat(0);
    if (NumGroups > 0)
    {
      std::stringstream aname;
      for (int jg = 0; jg < NumGroups; ++jg)
      {
        aname.str() = "";
        aname << jg << P.getGroupID(iat);
        if (J1UniqueFunctors.find(aname.str()) == J1UniqueFunctors.end())
        {
          aname.str() = "";
          aname << jg;
        }
        if (J1UniqueFunctors[aname.str()] != nullptr)
          curVat += J1UniqueFunctors[aname.str()]->evaluateV(-1, Ions.first(jg), Ions.last(jg), dist.data(),
                                                             DistCompressed.data());
      }
    }
    else
    {
      std::stringstream aname;
      for (int c = 0; c < Nions; ++c)
      {
        aname.str() = "";
        aname << Ions.getGroupID(c) << P.getGroupID(iat);
        if (J1UniqueFunctors.find(aname.str()) == J1UniqueFunctors.end())
        {
          aname.str() = "";
          aname << Ions.getGroupID(c);
        }
        if (J1UniqueFunctors[aname.str()] != nullptr)
          curVat += J1UniqueFunctors[aname.str()]->evaluate(dist[c]);
      }
    }
    return curVat;
  }

  void evaluateRatiosAlltoOne(ParticleSet& P, std::vector<ValueType>& ratios) override
  {
    const auto& dist = P.getDistTable(myTableID).getTempDists();
    curAt            = valT(0);
    if (NumGroups > 0)
    {
      for (int ig = 0; ig < NumGroups; ++ig)
      {
        for (int jg = 0; jg < NumTargetGroups; ++jg)
        {
          std::stringstream gid;
          gid << ig << jg;
          if (J1UniqueFunctors.find(gid.str()) == J1UniqueFunctors.end())
          {
            gid.str() = "";
            gid << ig;
          }

          if (J1UniqueFunctors[gid.str()] != nullptr)
            curAt += J1UniqueFunctors[gid.str()]->evaluateV(-1, Ions.first(ig), Ions.last(ig), dist.data(),
                                                            DistCompressed.data());
        }
      }
    }
    else
    {
      for (int ig = 0; ig < Nions; ++ig)
      {
        for (int jg = 0; jg < NumTargetGroups; ++jg)
        {
          std::stringstream gid;
          gid << Ions.getGroupID(ig) << jg;
          if (J1UniqueFunctors.find(gid.str()) == J1UniqueFunctors.end())
          {
            gid.str() = "";
            gid << Ions.getGroupID(ig);
          }
          if (J1UniqueFunctors[gid.str()] != nullptr)
            curAt += J1UniqueFunctors[gid.str()]->evaluate(dist[ig]);
        }
      }
    }

    for (int i = 0; i < Nelec; ++i)
      ratios[i] = std::exp(Vat[i] - curAt);
  }

  inline LogValueType evaluateGL(const ParticleSet& P,
                                 ParticleSet::ParticleGradient_t& G,
                                 ParticleSet::ParticleLaplacian_t& L,
                                 bool fromscratch = false) override
  {
    if (fromscratch)
      recompute(P);

    for (size_t iat = 0; iat < Nelec; ++iat)
      G[iat] += Grad[iat];
    for (size_t iat = 0; iat < Nelec; ++iat)
      L[iat] -= Lap[iat];
    return LogValue = -simd::accumulate_n(Vat.data(), Nelec, valT());
  }

  /** compute gradient and lap
   * @return lap
   */
  inline valT accumulateGL(const valT* restrict du, const valT* restrict d2u, const DisplRow& displ, posT& grad) const
  {
    valT lap(0);
    constexpr valT lapfac = OHMMS_DIM - RealType(1);
    //#pragma omp simd reduction(+:lap)
    for (int jat = 0; jat < Nions; ++jat)
      lap += d2u[jat] + lapfac * du[jat];
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      const valT* restrict dX = displ.data(idim);
      valT s                  = valT();
      //#pragma omp simd reduction(+:s)
      for (int jat = 0; jat < Nions; ++jat)
        s += du[jat] * dX[jat];
      grad[idim] = s;
    }
    return lap;
  }

  /** compute U, dU and d2U
   * @param P quantum particleset
   * @param iat the moving particle
   * @param dist starting address of the distances of the ions wrt the iat-th particle
   */
  inline void computeU3(const ParticleSet& P, int iat, const DistRow& dist)
  {
    if (NumGroups > 0)
    { //ions are grouped
      constexpr valT czero(0);
      std::fill_n(U.data(), Nions, czero);
      std::fill_n(dU.data(), Nions, czero);
      std::fill_n(d2U.data(), Nions, czero);

      for (int jg = 0; jg < NumGroups; ++jg)
      {
        std::stringstream gid;
        gid << jg << P.getGroupID(iat);
        if (J1UniqueFunctors.find(gid.str()) == J1UniqueFunctors.end())
        {
          gid.str() = "";
          gid << jg;
        }
        if (J1UniqueFunctors[gid.str()] == nullptr)
          continue;
        J1UniqueFunctors[gid.str()]->evaluateVGL(-1, Ions.first(jg), Ions.last(jg), dist.data(), U.data(), dU.data(),
                                                 d2U.data(), DistCompressed.data(), DistIndice.data());
      }
    }
    else
    {
      for (int c = 0; c < Nions; ++c)
      {
        std::stringstream gid;
        gid << Ions.getGroupID(c) << P.getGroupID(iat);
        if (J1UniqueFunctors.find(gid.str()) == J1UniqueFunctors.end())
        {
          gid.str() = "";
          gid << Ions.getGroupID(c);
        }
        if (J1UniqueFunctors[gid.str()] != nullptr)
        {
          U[c] = J1UniqueFunctors[gid.str()]->evaluate(dist[c], dU[c], d2U[c]);
          dU[c] /= dist[c];
        }
      }
    }
  }

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   */
  GradType evalGrad(ParticleSet& P, int iat) override { return GradType(Grad[iat]); }

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   *
   * Using getTempDists(). curAt, curGrad and curLap are computed.
   */
  PsiValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat) override
  {
    UpdateMode = ORB_PBYP_PARTIAL;

    computeU3(P, iat, P.getDistTable(myTableID).getTempDists());
    curLap = accumulateGL(dU.data(), d2U.data(), P.getDistTable(myTableID).getTempDispls(), curGrad);
    curAt  = simd::accumulate_n(U.data(), Nions, valT());
    grad_iat += curGrad;
    return std::exp(static_cast<PsiValueType>(Vat[iat] - curAt));
  }

  /** Rejected move. Nothing to do */
  inline void restore(int iat) override {}

  /** Accpted move. Update Vat[iat],Grad[iat] and Lap[iat] */
  void acceptMove(ParticleSet& P, int iat, bool safe_to_delay = false) override
  {
    if (UpdateMode == ORB_PBYP_RATIO)
    {
      computeU3(P, iat, P.getDistTable(myTableID).getTempDists());
      curLap = accumulateGL(dU.data(), d2U.data(), P.getDistTable(myTableID).getTempDispls(), curGrad);
    }

    LogValue += Vat[iat] - curAt;
    Vat[iat]  = curAt;
    Grad[iat] = curGrad;
    Lap[iat]  = curLap;
  }


  inline void registerData(ParticleSet& P, WFBufferType& buf) override
  {
    if (Bytes_in_WFBuffer == 0)
    {
      Bytes_in_WFBuffer = buf.current();
      buf.add(Vat.begin(), Vat.end());
      buf.add(Grad.begin(), Grad.end());
      buf.add(Lap.begin(), Lap.end());
      Bytes_in_WFBuffer = buf.current() - Bytes_in_WFBuffer;
      // free local space
      Vat.free();
      Grad.free();
      Lap.free();
    }
    else
    {
      buf.forward(Bytes_in_WFBuffer);
    }
  }

  inline LogValueType updateBuffer(ParticleSet& P, WFBufferType& buf, bool fromscratch = false) override
  {
    evaluateGL(P, P.G, P.L, false);
    buf.forward(Bytes_in_WFBuffer);
    return LogValue;
  }

  inline void copyFromBuffer(ParticleSet& P, WFBufferType& buf) override
  {
    Vat.attachReference(buf.lendReference<valT>(Nelec), Nelec);
    Grad.attachReference(buf.lendReference<posT>(Nelec), Nelec);
    Lap.attachReference(buf.lendReference<valT>(Nelec), Nelec);
  }

  inline void setVars(const opt_variables_type& vars)
  {
    NumVars = vars.size();
    if (NumVars == 0)
      return;
    myVars = vars;
    dLogPsi.resize(NumVars);
    gradLogPsi.resize(NumVars, 0);
    lapLogPsi.resize(NumVars, 0);
    for (int i = 0; i < NumVars; ++i)
    {
      gradLogPsi[i] = new WavefunctionFirstDerivativeType(Nelec);
      lapLogPsi[i]  = new WavefunctionSecondDerivativeType(Nelec);
    }
  }


  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet& tqp) const override
  {
    auto j1copy         = std::make_unique<J1SpinOrbitalSoA<FT>>(myName, Ions, tqp);
    j1copy->Optimizable = Optimizable;
    if (NumGroups > 0)
    {
      for (int i = 0; i < NumGroups; i++)
      {
        std::stringstream gid;
        gid << i;
        auto pos = J1UniqueFunctors.find(gid.str());
        if (pos != J1UniqueFunctors.end())
        {
          auto fc = std::make_unique<FT>(*pos->second.get());
          j1copy->addFunc(i, std::move(fc));
        }
        for (int j = 0; j < NumTargetGroups; j++)
        {
          std::stringstream gid;
          gid << i << j;
          auto pos = J1UniqueFunctors.find(gid.str());
          if (pos != J1UniqueFunctors.end())
          {
            auto fc = std::make_unique<FT>(*pos->second.get());
            j1copy->addFunc(i, std::move(fc), j);
          }
        }
      }
    }
    else
    {
      for (int i = 0; i < Nions; ++i)
      {
        std::stringstream gid;
        gid << i;
        auto pos = J1UniqueFunctors.find(gid.str());
        if (pos != J1UniqueFunctors.end())
        {
          auto fc = std::make_unique<FT>(*pos->second.get());
          j1copy->addFunc(i, std::move(fc));
        }
        for (int j = 0; j < NumTargetGroups; j++)
        {
          std::stringstream gid;
          gid << i << j;
          auto pos = J1UniqueFunctors.find(gid.str());
          if (pos != J1UniqueFunctors.end())
          {
            auto fc = std::make_unique<FT>(*pos->second.get());
            j1copy->addFunc(i, std::move(fc), j);
          }
        }
      }
    }
    j1copy->setVars(myVars);
    j1copy->OffSet = OffSet;
    return j1copy;
  }

  /**@{ WaveFunctionComponent virtual functions that are not essential for the development */
  void reportStatus(std::ostream& os) override
  {
    for (auto& J1UniqueFunctorsptrpair : J1UniqueFunctors)
      if (J1UniqueFunctorsptrpair.second != nullptr)
        J1UniqueFunctorsptrpair.second->myVars.print(os);
  }

  void checkInVariables(opt_variables_type& active) override
  {
    myVars.clear();
    for (auto& J1UniqueFunctorsptrpair : J1UniqueFunctors)
    {
      if (J1UniqueFunctorsptrpair.second != nullptr)
      {
        J1UniqueFunctorsptrpair.second->checkInVariables(active);
        J1UniqueFunctorsptrpair.second->checkInVariables(myVars);
      }
    }
  }
  void checkOutVariables(const opt_variables_type& active) override
  {
    myVars.clear();
    for (auto& J1UniqueFunctorsptrpair : J1UniqueFunctors)
    {
      if (J1UniqueFunctorsptrpair.second)
      {
        J1UniqueFunctorsptrpair.second->myVars.getIndex(active);
        myVars.insertFrom(J1UniqueFunctorsptrpair.second->myVars);
      }
    }
    myVars.getIndex(active);
    NumVars = myVars.size();
    myVars.print(std::cout);
    if (NumVars && dLogPsi.size() == 0)
    {
      dLogPsi.resize(NumVars);
      gradLogPsi.resize(NumVars, 0);
      lapLogPsi.resize(NumVars, 0);
      for (int i = 0; i < NumVars; ++i)
      {
        gradLogPsi[i] = new WavefunctionFirstDerivativeType(Nelec);
        lapLogPsi[i]  = new WavefunctionSecondDerivativeType(Nelec);
      }
      OffSet.resize(J1Functors.size());
      // Find first active variable for the starting offset
      int varoffset = -1;
      for (int i = 0; i < myVars.size(); i++)
      {
        varoffset = myVars.Index[i];
        if (varoffset != -1)
          break;
      }

      for (int i = 0; i < J1Functors.size(); ++i)
      {
        if (J1Functors[i] && J1Functors[i]->myVars.Index.size())
        {
          OffSet[i].first  = J1Functors[i]->myVars.Index.front() - varoffset;
          OffSet[i].second = J1Functors[i]->myVars.Index.size() + OffSet[i].first;
        }
        else
        {
          OffSet[i].first = OffSet[i].second = -1;
        }
      }
    }
    Optimizable = myVars.is_optimizable();
    for (auto& J1UniqueFunctorsptrpair : J1UniqueFunctors)
      if (J1UniqueFunctorsptrpair.second != nullptr)
        J1UniqueFunctorsptrpair.second->checkOutVariables(active);
  }

  void resetParameters(const opt_variables_type& active) override
  {
    if (!Optimizable)
      return;
    for (auto& J1UniqueFunctorsptrpair : J1UniqueFunctors)
      if (J1UniqueFunctorsptrpair.second != nullptr)
        J1UniqueFunctorsptrpair.second->resetParameters(active);

    for (int i = 0; i < myVars.size(); ++i)
    {
      int ii = myVars.Index[i];
      if (ii >= 0)
        myVars[i] = active[ii];
    }
  }
  /**@} */

  inline GradType evalGradSource(ParticleSet& P, ParticleSet& source, int isrc) override
  {
    GradType g_return(0.0);
    const DistanceTableData& d_ie(P.getDistTable(myTableID));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      const auto& dist  = d_ie.getDistRow(iat);
      const auto& displ = d_ie.getDisplRow(iat);
      RealType r        = dist[isrc];
      RealType rinv     = 1.0 / r;
      PosType dr        = displ[isrc];
      std::stringstream gid;
      gid << source.getGroupID(isrc) << P.getGroupID(iat);
      if (J1UniqueFunctors.find(gid.str()) == J1UniqueFunctors.end())
      {
        gid.str() = "";
        gid << source.getGroupID(isrc);
      }

      if (J1UniqueFunctors[gid.str()] != nullptr)
      {
        U[isrc] = J1UniqueFunctors[gid.str()]->evaluate(dist[isrc], dU[isrc], d2U[isrc], d3U[isrc]);
        g_return -= dU[isrc] * rinv * dr;
      }
    }
    return g_return;
  }

  inline GradType evalGradSource(ParticleSet& P,
                                 ParticleSet& source,
                                 int isrc,
                                 TinyVector<ParticleSet::ParticleGradient_t, OHMMS_DIM>& grad_grad,
                                 TinyVector<ParticleSet::ParticleLaplacian_t, OHMMS_DIM>& lapl_grad) override
  {
    GradType g_return(0.0);
    const DistanceTableData& d_ie(P.getDistTable(myTableID));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      const auto& dist  = d_ie.getDistRow(iat);
      const auto& displ = d_ie.getDisplRow(iat);
      RealType r        = dist[isrc];
      RealType rinv     = 1.0 / r;
      PosType dr        = displ[isrc];
      std::stringstream gid;
      gid << source.getGroupID(isrc) << P.getGroupID(iat);
      if (J1UniqueFunctors.find(gid.str()) == J1UniqueFunctors.end())
      {
        gid.str() = "";
        gid << source.getGroupID(isrc);
      }

      if (J1UniqueFunctors[gid.str()] != nullptr)
      {
        U[isrc] = J1UniqueFunctors[gid.str()]->evaluate(dist[isrc], dU[isrc], d2U[isrc], d3U[isrc]);
      }
      else
      {
        APP_ABORT("J1OrbitalSoa::evaluateGradSource:  J1UniqueFunctors[gid]==nullptr")
      }

      g_return -= dU[isrc] * rinv * dr;

      //The following terms depend only on the radial component r.  Thus,
      //we compute them and mix with position vectors to acquire the full
      //cartesian vector objects.
      valT grad_component = (d2U[isrc] - dU[isrc] * rinv);
      valT lapl_component = d3U[isrc] + 2 * rinv * grad_component;

      for (int idim = 0; idim < OHMMS_DIM; idim++)
      {
        grad_grad[idim][iat] += dr[idim] * dr * rinv * rinv * grad_component;
        grad_grad[idim][iat][idim] += rinv * dU[isrc];

        lapl_grad[idim][iat] -= lapl_component * rinv * dr[idim];
      }
    }
    return g_return;
  }
};


} // namespace qmcplusplus
#endif
