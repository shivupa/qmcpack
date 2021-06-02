//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_ONEBODYSPINJASTROW_OPTIMIZED_SOA_H
#define QMCPLUSPLUS_ONEBODYSPINJASTROW_OPTIMIZED_SOA_H
#include "Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "QMCWaveFunctions/Jastrow/DiffOneBodyJastrowOrbital.h"
#include "Utilities/qmc_common.h"
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
  int Nelec;
  ///number of groups for the sources, e.g., for the atomic centers
  int NumSourceGroups;
  ///number of groupsfor the targets, e.g., for the up/down electrons
  int NumTargetGroups;
  ///reference to the sources (ions)
  const ParticleSet& Ions;
  ///reference to the targets (electrons)
  ParticleSet& Elecs;

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
  ///Container for \f$F[ig*NumGroups+jg]\f$
  std::vector<FT*> F;
  ///Unique J1 set
  std::map<std::string, std::unique_ptr<FT>> J1Unique;

  J1SpinOrbitalSoA(const std::string& obj_name, const ParticleSet& ions, ParticleSet& els)
      : WaveFunctionComponent("J1SpinOrbitalSoA", obj_name), myTableID(els.addTable(ions)), Ions(ions), Elecs(els)
  {
    if (myName.empty())
      throw std::runtime_error("J1SpinOrbitalSoA object name cannot be empty!");
    initialize(els);
  }

  J1SpinOrbitalSoA(const J1SpinOrbitalSoA& rhs) = delete;

  ~J1SpinOrbitalSoA()
  {
    for (int i = 0; i < F.size(); ++i)
      if (F[i] != nullptr)
        delete F[i];
  }

  /* initialize storage */
  void initialize(const ParticleSet& els)
  {
    Nions           = Ions.getTotalNum();
    Nelec           = Elecs.getTotalNum();
    NumSourceGroups = Ions.groups();
    NumTargetGroups = Elecs.groups();
    F.resize(NumSourceGroups * NumTargetGroups, nullptr);
    //if (NumSourceGroups > 1 && !Ions.IsGrouped) is this necessary?
    //{
    //  NumSourceGroups = 0;
    //}
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

  void addFunc(int source_type, FT* afunc, int target_type = -1)
  {
    // make all pair terms for a certain atom equal to first atom - first specified target initially in case some terms are not provided explicitly
    // e.g. if atomA-spin1 specified first in input and atomA-spin0 and atomA-spin(2-...) will all equal atomA-spin1 (unless provided explicitly)
    for (int j = 0; j < NumTargetGroups; ++j)
      if (F[source_type * NumTargetGroups + j] == nullptr)
        F[source_type * NumTargetGroups + j] = afunc;
    std::stringstream aname;
    aname << source_type << target_type;
    J1Unique[aname.str()] = std::unique_ptr<FT>(afunc);
    //if (F[source_type] != nullptr)
    //  delete F[source_type];
    F[source_type * NumTargetGroups + target_type] = afunc;
  }

  void recompute(const ParticleSet& P)
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
                           ParticleSet::ParticleLaplacian_t& L)
  {
    return evaluateGL(P, G, L, true);
  }

  void evaluateHessian(ParticleSet& P, HessVector_t& grad_grad_psi)
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
        int gid    = Ions.GroupID[iat] * NumTargetGroups + Elecs.GroupID[iel];
        auto* func = F[gid];
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

  PsiValueType ratio(ParticleSet& P, int iat)
  {
    UpdateMode = ORB_PBYP_RATIO;
    curAt      = computeU(P, iat, P.getDistTable(myTableID).getTempDists());
    return std::exp(static_cast<PsiValueType>(Vat[iat] - curAt));
  }

  inline void evaluateRatios(const VirtualParticleSet& VP, std::vector<ValueType>& ratios)
  {
    for (int k = 0; k < ratios.size(); ++k)
      ratios[k] = std::exp(Vat[VP.refPtcl] - computeU(VP.refPS, VP.refPtcl, VP.getDistTable(myTableID).getDistRow(k)));
  }

  inline valT computeU(const ParticleSet& P, int iat, const DistRow& dist)
  {
    valT curUat(0);
    const int igt = P.GroupID[iat] * NumTargetGroups;
    for (int jg = 0; jg < NumTargetGroups; ++jg)
    {
      const FuncType& f1(*F[igt + jg]);
      int iStart = P.first(jg);
      int iEnd   = P.last(jg);
      curUat += f1.evaluateV(iat, iStart, iEnd, dist.data(), DistCompressed.data());
    }
    return curUat;
  }

  void evaluateRatiosAlltoOne(ParticleSet& P, std::vector<ValueType>& ratios)
  {
    const auto& d_table = P.getDistTable(myTableID);
    const auto& dist    = d_table.getTempDists();

    for (int ig = 0; ig < NumSourceGroups; ++ig)
    {
      const int igt = ig * NumTargetGroups;
      valT sumU(0);
      for (int jg = 0; jg < NumTargetGroups; ++jg)
      {
        const FuncType& f1(*F[igt + jg]);
        int iStart = P.first(jg);
        int iEnd   = P.last(jg);
        sumU += f1.evaluateV(-1, iStart, iEnd, dist.data(), DistCompressed.data());
      }

      for (int i = P.first(ig); i < P.last(ig); ++i)
      {
        // remove self-interaction
        const valT Uself = F[igt + ig]->evaluate(dist[i]);
        ratios[i]        = std::exp(Vat[i] + Uself - sumU);
      }
    }
  }

  inline LogValueType evaluateGL(const ParticleSet& P,
                                 ParticleSet::ParticleGradient_t& G,
                                 ParticleSet::ParticleLaplacian_t& L,
                                 bool fromscratch = false)
  {
    if (fromscratch)
      recompute(P);

    for (size_t iat = 0; iat < Nelec; ++iat)
    {
      G[iat] += Grad[iat];
      L[iat] -= Lap[iat];
    }
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
    constexpr valT czero(0);
    std::fill_n(U.data(), Nions, czero);
    std::fill_n(dU.data(), Nions, czero);
    std::fill_n(d2U.data(), Nions, czero);

    const int igt = P.GroupID[iat] * NumTargetGroups;
    for (int jg = 0; jg < NumTargetGroups; ++jg)
    {
      const FuncType& f1(*F[igt + jg]);
      int iStart = P.first(jg);
      int iEnd   = std::min(Nions, P.last(jg));
      app_log() << "iStart  iEnd  iat" << iStart << " "  << iEnd << " " << iat << std::endl;
      f1.evaluateVGL(iat, iStart, iEnd, dist.data(), U.data(), dU.data(), d2U.data(), DistCompressed.data(),
                     DistIndice.data());
      app_log() << "OK" << std::endl;
    }
  }

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   */
  GradType evalGrad(ParticleSet& P, int iat) { return GradType(Grad[iat]); }

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   *
   * Using getTempDists(). curAt, curGrad and curLap are computed.
   */
  PsiValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat)
  {
    UpdateMode = ORB_PBYP_PARTIAL;

    computeU3(P, iat, P.getDistTable(myTableID).getTempDists());
    curLap = accumulateGL(dU.data(), d2U.data(), P.getDistTable(myTableID).getTempDispls(), curGrad);
    curAt  = simd::accumulate_n(U.data(), Nions, valT());
    grad_iat += curGrad;
    return std::exp(static_cast<PsiValueType>(Vat[iat] - curAt));
  }

  /** Rejected move. Nothing to do */
  inline void restore(int iat) {}

  /** Accpted move. Update Vat[iat],Grad[iat] and Lap[iat] */
  void acceptMove(ParticleSet& P, int iat, bool safe_to_delay = false)
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


  inline void registerData(ParticleSet& P, WFBufferType& buf)
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

  inline LogValueType updateBuffer(ParticleSet& P, WFBufferType& buf, bool fromscratch = false)
  {
    evaluateGL(P, P.G, P.L, false);
    buf.forward(Bytes_in_WFBuffer);
    return LogValue;
  }

  inline void copyFromBuffer(ParticleSet& P, WFBufferType& buf)
  {
    Vat.attachReference(buf.lendReference<valT>(Nelec), Nelec);
    Grad.attachReference(buf.lendReference<posT>(Nelec), Nelec);
    Lap.attachReference(buf.lendReference<valT>(Nelec), Nelec);
  }

  WaveFunctionComponentPtr makeClone(ParticleSet& tqp) const
  {
    J1SpinOrbitalSoA<FT>* j1copy = new J1SpinOrbitalSoA<FT>(myName, Ions, tqp);
    j1copy->Optimizable          = Optimizable;
    for (size_t i = 0, n = F.size(); i < n; ++i)
    {
      if (F[i] != nullptr)
        j1copy->addFunc(i, new FT(*F[i]));
    }
    if (dPsi)
    {
      j1copy->dPsi = dPsi->makeClone(tqp);
    }
    return j1copy;
  }

  /**@{ WaveFunctionComponent virtual functions that are not essential for the development */
  void reportStatus(std::ostream& os)
  {
    for (size_t i = 0, n = F.size(); i < n; ++i)
    {
      if (F[i] != nullptr)
        F[i]->myVars.print(os);
    }
  }

  void checkInVariables(opt_variables_type& active)
  {
    myVars.clear();
    auto it(J1Unique.begin()), it_end(J1Unique.end());
    while (it != it_end)
    {
      (*it).second->checkInVariables(active);
      (*it).second->checkInVariables(myVars);
      ++it;
    }
  }
  void checkOutVariables(const opt_variables_type& active)
  {
    myVars.getIndex(active);
    Optimizable = myVars.is_optimizable();
    auto it(J1Unique.begin()), it_end(J1Unique.end());
    while (it != it_end)
    {
      (*it).second->checkOutVariables(active);
      ++it;
    }
    if (dPsi)
      dPsi->checkOutVariables(active);
  }

  void resetParameters(const opt_variables_type& active)
  {
    if (!Optimizable)
      return;
    auto it(J1Unique.begin()), it_end(J1Unique.end());
    while (it != it_end)
    {
      (*it).second->resetParameters(active);
      ++it;
    }
    if (dPsi)
      dPsi->resetParameters(active);
    for (int i = 0; i < myVars.size(); ++i)
    {
      int ii = myVars.Index[i];
      if (ii >= 0)
       myVars[i] = active[ii];
    }
  }
  /**@} */

  inline GradType evalGradSource(ParticleSet& P, ParticleSet& source, int isrc)
  {
    GradType g_return(0.0);
    const DistanceTableData& d_ie(P.getDistTable(myTableID));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      const auto& dist  = d_ie.getDistRow(iat);
      const auto& displ = d_ie.getDisplRow(iat);
      int gid           = source.GroupID[isrc] * NumTargetGroups + P.GroupID[iat];
      RealType r        = dist[isrc];
      RealType rinv     = 1.0 / r;
      PosType dr        = displ[isrc];

      if (F[gid] != nullptr)
      {
        U[isrc] = F[gid]->evaluate(dist[isrc], dU[isrc], d2U[isrc], d3U[isrc]);
        g_return -= dU[isrc] * rinv * dr;
      }
    }
    return g_return;
  }

  inline GradType evalGradSource(ParticleSet& P,
                                 ParticleSet& source,
                                 int isrc,
                                 TinyVector<ParticleSet::ParticleGradient_t, OHMMS_DIM>& grad_grad,
                                 TinyVector<ParticleSet::ParticleLaplacian_t, OHMMS_DIM>& lapl_grad)
  {
    GradType g_return(0.0);
    const DistanceTableData& d_ie(P.getDistTable(myTableID));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      const auto& dist  = d_ie.getDistRow(iat);
      const auto& displ = d_ie.getDisplRow(iat);
      int gid           = source.GroupID[isrc] * NumTargetGroups + P.GroupID[iat];
      RealType r        = dist[isrc];
      RealType rinv     = 1.0 / r;
      PosType dr        = displ[isrc];

      if (F[gid] != nullptr)
      {
        U[isrc] = F[gid]->evaluate(dist[isrc], dU[isrc], d2U[isrc], d3U[isrc]);
      }
      else
      {
        APP_ABORT("J1OrbitalSoa::evaluateGradSource:  F[gid]==nullptr")
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
