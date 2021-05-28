//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_DIFFERENTIAL_ONEBODYSPINJASTROW_H
#define QMCPLUSPLUS_DIFFERENTIAL_ONEBODYSPINJASTROW_H
#include "Configuration.h"
#include "QMCWaveFunctions/DiffWaveFunctionComponent.h"
#include "Particle/DistanceTableData.h"
#include "ParticleBase/ParticleAttribOps.h"
#include "Utilities/IteratorUtility.h"


namespace qmcplusplus
{
/** @ingroup WaveFunctionComponent
 *  @brief Specialization for two-body Jastrow function using multiple functors
 */
template<class FT>
class DiffOneBodySpinJastrowOrbital : public DiffWaveFunctionComponent
{
  ///number of variables this object handles
  int NumVars;
  ///number of target particles
  int NumPtcls;
  ///number of groups for the sources, e.g., for the atomic centers
  int NumSourceGroups;
  ///number of groupsfor the targets, e.g., for the up/down electrons
  int NumTargetGroups;
  ///index of the table
  const int myTableIndex;
  ///reference to the ions
  const ParticleSet& CenterRef;
  ///variables handled by this orbital
  opt_variables_type myVars;
  ///container for the Jastrow functions  for all the pairs
  std::vector<FT*> F;
  ///container for the unique Jastrow functions
  std::map<std::string, FT*> J1Unique;
  std::vector<std::pair<int, int>> OffSet;
  Vector<RealType> dLogPsi;
  std::vector<GradVectorType*> gradLogPsi;
  std::vector<ValueVectorType*> lapLogPsi;

public:
  ///constructor
  DiffOneBodySpinJastrowOrbital(const ParticleSet& centers, ParticleSet& els)
      : NumVars(0), myTableIndex(els.addTable(centers)), CenterRef(centers)
  {
    NumPtcls        = els.getTotalNum();
    NumTargetGroups = els.groups();
    NumSourceGroups = CenterRef.groups();
    F.resize(NumSourceGroups * NumTargetGroups, 0);
  }

  ~DiffOneBodySpinJastrowOrbital()
  {
    delete_iter(gradLogPsi.begin(), gradLogPsi.end());
    delete_iter(lapLogPsi.begin(), lapLogPsi.end());
  }

  /** Add a radial functor for a group
   * @param source_type group index of the center species
   * @param afunc radial functor
   */
  void addFunc(int source_type, FT* afunc, int target_type = -1)
  {
    //   S0    S1   S2
    // A A-S0  A-S1 A-S2
    // B B-S0  B-S1 B-S2

    // make all pair terms for a certain atom equal to first atom - first specified target initially in case some terms are not provided explicitly
    // e.g. if atomA-spin1 specified first in input and atomA-spin0 and atomA-spin(2-...) will all equal atomA-spin1 (unless provided explicitly)
    for (int j = 0; j < NumTargetGroups; ++j)
      if (F[source_type * NumTargetGroups + j] == nullptr)
        F[source_type * NumTargetGroups + j] = afunc;
    F[source_type * NumTargetGroups + target_type] = afunc;
    std::stringstream aname;
    aname << source_type << target_type;
    J1Unique[aname.str()] = afunc;
  }


  ///reset the value of all the unique Two-Body Jastrow functions
  void resetParameters(const opt_variables_type& active)
  {
    typename std::map<std::string, FT*>::iterator it(J1Unique.begin()), it_end(J1Unique.end());
    while (it != it_end)
    {
      (*it++).second->resetParameters(active);
    }
  }

  void checkOutVariables(const opt_variables_type& active)
  {
    myVars.clear();
    typename std::map<std::string, FT*>::iterator it(J1Unique.begin()), it_end(J1Unique.end());
    while (it != it_end)
    {
      (*it).second->myVars.getIndex(active);
      myVars.insertFrom((*it).second->myVars);
      ++it;
    }
    myVars.removeInactive();

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
        gradLogPsi[i] = new GradVectorType(NumPtcls);
        lapLogPsi[i]  = new ValueVectorType(NumPtcls);
      }
      OffSet.resize(F.size());
      // Find first active variable for the starting offset
      int varoffset = -1;
      for (int i = 0; i < myVars.size(); i++)
      {
        varoffset = myVars.Index[i];
        if (varoffset != -1)
          break;
      }
      for (int i = 0; i < F.size(); ++i)
      {
        if (F[i] && F[i]->myVars.Index.size())
        {
          OffSet[i].first  = F[i]->myVars.Index.front() - varoffset;
          OffSet[i].second = F[i]->myVars.Index.size() + OffSet[i].first;
        }
        else
        {
          OffSet[i].first = OffSet[i].second = -1;
        }
      }
    }
  }

  void evaluateDerivatives(ParticleSet& P,
                           const opt_variables_type& active,
                           std::vector<ValueType>& dlogpsi,
                           std::vector<ValueType>& dhpsioverpsi)
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

  void evaluateDerivativesWF(ParticleSet& P, const opt_variables_type& active, std::vector<ValueType>& dlogpsi)
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
      const auto& d_table = P.getDistTable(myTableIndex);
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
        FT* func = F[i];
        if (func == 0)
          continue;
        int first(OffSet[i].first);
        int last(OffSet[i].second);
        bool recalcFunc(false);
        for (int rcs = first; rcs < last; rcs++)
          if (rcsingles[rcs] == true)
            recalcFunc = true;
        if (recalcFunc)
        {
          size_t nn = d_table.get_neighbors(i, func->cutoff_radius, iadj.data(), dist.data(), displ.data());
          for (size_t nj = 0; nj < nn; ++nj)
          {
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
      gradLogPsi[i] = new GradVectorType(NumPtcls);
      lapLogPsi[i]  = new ValueVectorType(NumPtcls);
    }
  }

  DiffWaveFunctionComponentPtr makeClone(ParticleSet& tqp) const
  {
    DiffOneBodySpinJastrowOrbital<FT>* j1copy = new DiffOneBodySpinJastrowOrbital<FT>(CenterRef, tqp);
    std::map<const FT*, FT*> fcmap;
    for (int ig = 0; ig < NumSourceGroups; ++ig)
      for (int jg = ig; jg < NumTargetGroups; ++jg)
      {
        int ij = ig * NumTargetGroups + jg;
        if (F[ij] == 0)
          continue;
        typename std::map<const FT*, FT*>::iterator fit = fcmap.find(F[ij]);
        if (fit == fcmap.end())
        {
          FT* fc = new FT(*F[ij]);
          j1copy->addFunc(ig, fc, jg);
          fcmap[F[ij]] = fc;
        }
      }
    j1copy->myVars.clear();
    j1copy->myVars.insertFrom(myVars);
    j1copy->NumVars         = NumVars;
    j1copy->NumPtcls        = NumPtcls;
    j1copy->NumSourceGroups = NumSourceGroups;
    j1copy->NumTargetGroups = NumTargetGroups;
    j1copy->dLogPsi.resize(NumVars);
    j1copy->gradLogPsi.resize(NumVars, 0);
    j1copy->lapLogPsi.resize(NumVars, 0);
    for (int i = 0; i < NumVars; ++i)
    {
      j1copy->gradLogPsi[i] = new GradVectorType(NumPtcls);
      j1copy->lapLogPsi[i]  = new ValueVectorType(NumPtcls);
    }
    j1copy->OffSet = OffSet;
    return j1copy;
  }
};


} // namespace qmcplusplus
#endif
