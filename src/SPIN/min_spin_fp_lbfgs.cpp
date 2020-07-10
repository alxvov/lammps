/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------
   Contributing authors: Aleksei Ivanov (University of Iceland)
                         Julien Tranchida (SNL)

   Please cite the related publication:
   Ivanov, A. V., Uzdin, V. M., & Jónsson, H. (2019). Fast and Robust
   Algorithm for the Minimisation of the Energy of Spin Systems. arXiv
   preprint arXiv:1904.02669.
------------------------------------------------------------------------- */

#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "min_spin_fp_lbfgs.h"
#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "force.h"
#include "update.h"
#include "output.h"
#include "timer.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "math_special.h"
#include "math_const.h"
#include "universe.h"

using namespace LAMMPS_NS;
using namespace MathConst;

static const char cite_minstyle_spin_fp_lbfgs[] =
  "min_style spin/fp_lbfgs command:\n\n"
  "@article{ivanov2019fast,\n"
  "title={Fast and Robust Algorithm for the Minimisation of the Energy of "
  "Spin Systems},\n"
  "author={Ivanov, A. V and Uzdin, V. M. and J{\'o}nsson, H.},\n"
  "journal={arXiv preprint arXiv:1904.02669},\n"
  "year={2019}\n"
  "}\n\n";

// EPS_ENERGY = minimum normalization for energy tolerance

#define EPS_ENERGY 1.0e-8

#define DELAYSTEP 5

/* ---------------------------------------------------------------------- */

MinSpinFP_LBFGS::MinSpinFP_LBFGS(LAMMPS *lmp) :
  Min(lmp), g_old(NULL), g_cur(NULL), p_s(NULL), rho(NULL),  alpha(NULL), ds(NULL), dy(NULL), sp_copy(NULL)
{
  if (lmp->citeme) lmp->citeme->add(cite_minstyle_spin_fp_lbfgs);
  nlocal_max = 0;

  // nreplica = number of partitions
  // ireplica = which world I am in universe

  nreplica = universe->nworlds;
  ireplica = universe->iworld;
  use_line_search = 1;  // line search as default option for LBFGS
  maxepsrot = MY_2PI / (100.0);
  num_mem = 5;
  intervalsize=100.0;
}

/* ---------------------------------------------------------------------- */

MinSpinFP_LBFGS::~MinSpinFP_LBFGS()
{
    memory->destroy(g_old);
    memory->destroy(g_cur);
    memory->destroy(p_s);
    memory->destroy(ds);
    memory->destroy(dy);
    memory->destroy(rho);
    memory->destroy(alpha);
    if (use_line_search)
      memory->destroy(sp_copy);
}

/* ---------------------------------------------------------------------- */

void MinSpinFP_LBFGS::init()
{
  local_iter = 0;
  der_e_cur = 0.0;
  der_e_pr = 0.0;
  e_cur = 0.0;
  e_pr = 0.0;

  c1=1.0e-4;
  c2=0.9;
  maxiterls=10;
  epsdx=1.0e-10;

  Min::init();

  if (linestyle == 4) use_line_search = 0;

  // warning if line_search combined to gneb

  if ((nreplica >= 1) && (linestyle != 4) && (comm->me == 0))
    error->warning(FLERR,"Line search incompatible gneb");

  // set back use_line_search to 0 if more than one replica
  if (nreplica > 1){
    use_line_search = 0;
  }

  last_negative = update->ntimestep;

  // allocate tables

  nlocal_max = atom->nlocal;
  memory->grow(g_old,3*nlocal_max,"min/spin/fp_lbfgs:g_old");
  memory->grow(g_cur,3*nlocal_max,"min/spin/fp_lbfgs:g_cur");
  memory->grow(p_s,3*nlocal_max,"min/spin/fp_lbfgs:p_s");
  memory->grow(rho,num_mem,"min/spin/fp_lbfgs:rho");
  memory->grow(alpha,num_mem,"min/spin/fp_lbfgs:alpha");
  memory->grow(ds,num_mem,3*nlocal_max,"min/spin/fp_lbfgs:ds");
  memory->grow(dy,num_mem,3*nlocal_max,"min/spin/fp_lbfgs:dy");
  if (use_line_search)
    memory->grow(sp_copy,nlocal_max,3,"min/spin/fp_lbfgs:sp_copy");

}

/* ---------------------------------------------------------------------- */

void MinSpinFP_LBFGS::setup_style()
{
  double **v = atom->v;
  int nlocal = atom->nlocal;

  // check if the atom/spin style is defined

  if (!atom->sp_flag)
    error->all(FLERR,"min spin/fp_lbfgs requires atom/spin style");

  for (int i = 0; i < nlocal; i++)
    v[i][0] = v[i][1] = v[i][2] = 0.0;
}

/* ---------------------------------------------------------------------- */

int MinSpinFP_LBFGS::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"discrete_factor") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    double discrete_factor;
    discrete_factor = force->numeric(FLERR,arg[1]);
    maxepsrot = MY_2PI / (10 * discrete_factor);
    return 2;
  }
  if (strcmp(arg[0],"memory") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    num_mem = force->numeric(FLERR,arg[1]);
    return 2;
  }
  if (strcmp(arg[0],"intervalsize") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    intervalsize = force->numeric(FLERR,arg[1]);
    return 2;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   set current vector lengths and pointers
   called after atoms have migrated
------------------------------------------------------------------------- */

void MinSpinFP_LBFGS::reset_vectors()
{
  // atomic dof

  // size sp is 4N vector
  nvec = 4 * atom->nlocal;
  if (nvec) spvec = atom->sp[0];

  nvec = 3 * atom->nlocal;
  if (nvec) fmvec = atom->fm[0];

  if (nvec) xvec = atom->x[0];
  if (nvec) fvec = atom->f[0];
}

/* ----------------------------------------------------------------------
   minimization via damped spin dynamics
------------------------------------------------------------------------- */

int MinSpinFP_LBFGS::iterate(int maxiter)
{
  int nlocal = atom->nlocal;
  bigint ntimestep;
  double fmdotfm,fmsq;
  int flag, flagall;
  double **sp = atom->sp;
  double der_e_cur_tmp = 0.0;

  if (nlocal_max < nlocal) {
    nlocal_max = nlocal;
    local_iter = 0;
    memory->grow(g_old,3*nlocal_max,"min/spin/fp_lbfgs:g_old");
    memory->grow(g_cur,3*nlocal_max,"min/spin/fp_lbfgs:g_cur");
    memory->grow(p_s,3*nlocal_max,"min/spin/fp_lbfgs:p_s");
    memory->grow(rho,num_mem,"min/spin/fp_lbfgs:rho");
    memory->grow(alpha,num_mem,"min/spin/fp_lbfgs:alpha");
    memory->grow(ds,num_mem,3*nlocal_max,"min/spin/fp_lbfgs:ds");
    memory->grow(dy,num_mem,3*nlocal_max,"min/spin/fp_lbfgs:dy");
    if (use_line_search)
      memory->grow(sp_copy,nlocal_max,3,"min/spin/fp_lbfgs:sp_copy");
  }

  for (int iter = 0; iter < maxiter; iter++) {

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++update->ntimestep;
    niter++;

    // optimize timestep across processes / replicas
    // need a force calculation for timestep optimization

    if (use_line_search) {

      // here we need to do line search
      if (local_iter == 0){
        eprevious = ecurrent;
        ecurrent = energy_force(0);
        calc_gradient();
        e_cur = ecurrent;
      }

      calc_search_direction();
      der_e_cur = 0.0;
      for (int i = 0; i < 3 * nlocal; i++)
        der_e_cur += g_cur[i] * p_s[i];
      MPI_Allreduce(&der_e_cur,&der_e_cur_tmp,1,MPI_DOUBLE,MPI_SUM,world);
      der_e_cur = der_e_cur_tmp;
      if (update->multireplica == 1) {
        MPI_Allreduce(&der_e_cur_tmp,&der_e_cur,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      }
      for (int i = 0; i < nlocal; i++)
        for (int j = 0; j < 3; j++)
      sp_copy[i][j] = sp[i][j];

      eprevious = ecurrent;
      e_pr = e_cur;
      der_e_pr = der_e_cur;
      calc_and_make_step();
    }
    else{

      // here we don't do line search
      // but use cutoff rotation angle
      // if gneb calc., nreplica > 1
      // then calculate gradients and advance spins
      // of intermediate replicas only
      eprevious = ecurrent;
      ecurrent = energy_force(0);
      calc_gradient();
      calc_search_direction();
      advance_spins();
      neval++;
    }

    // energy tolerance criterion
    // only check after DELAYSTEP elapsed since velocties reset to 0
    // sync across replicas if running multi-replica minimization

    if (update->etol > 0.0 && ntimestep-last_negative > DELAYSTEP) {
      if (update->multireplica == 0) {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          return ETOL;
      } else {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return ETOL;
      }
    }

    // magnetic torque tolerance criterion
    // sync across replicas if running multi-replica minimization

    fmdotfm = fmsq = 0.0;
    if (update->ftol > 0.0) {
      if (normstyle == MAX) fmsq = max_torque();        // max torque norm
      else if (normstyle == INF) fmsq = inf_torque();   // inf torque norm
      else if (normstyle == TWO) fmsq = total_torque(); // Euclidean torque 2-norm
      else error->all(FLERR,"Illegal min_modify command");
      fmdotfm = fmsq*fmsq;
      if (update->multireplica == 0) {
        if (fmdotfm < update->ftol*update->ftol) return FTOL;
      } else {
        if (fmdotfm < update->ftol*update->ftol) flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return FTOL;
      }
    }

    // output for thermo, dump, restart files

    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}

/* ----------------------------------------------------------------------
   calculate gradients
---------------------------------------------------------------------- */

void MinSpinFP_LBFGS::calc_gradient()
{
  int nlocal = atom->nlocal;
  double **sp = atom->sp;
  double **fm = atom->fm;
  double hbar = force->hplanck/MY_2PI;

  double dot_prod = 0.0;
  // loop on all spins on proc.

  for (int i = 0; i < nlocal; i++) {
    dot_prod = 0.0;
    for (int cc = 0; cc < 3; cc++)
      dot_prod += fm[i][cc]*sp[i][cc];
    for (int cc = 0; cc < 3; cc++)
      g_cur[3 * i + cc] = -(fm[i][cc] - dot_prod*sp[i][cc]) * hbar;
  }
}

/* ----------------------------------------------------------------------
   search direction:
   Limited-memory BFGS.
   See Jorge Nocedal and Stephen J. Wright 'Numerical
   Optimization' Second Edition, 2006 (p. 177)
---------------------------------------------------------------------- */

void MinSpinFP_LBFGS::calc_search_direction()
{
  int nlocal = atom->nlocal;

  double dyds = 0.0;
  double sq = 0.0;
  double yy = 0.0;
  double yr = 0.0;
  double beta = 0.0;

  double dyds_global = 0.0;
  double sq_global = 0.0;
  double yy_global = 0.0;
  double yr_global = 0.0;

  int m_index = local_iter % num_mem; // memory index
  int c_ind = 0;

  double factor;
  double scaling = 1.0;

  // for multiple replica do not move end points
  if (nreplica > 1) {
    if (ireplica == 0 || ireplica == nreplica - 1) {
      factor = 0.0;
    }
    else factor = 1.0;
  }else{
    factor = 1.0;
  }

  if (local_iter == 0){         // steepest descent direction

    //if no line search then calculate maximum rotation
    if (use_line_search == 0)
      scaling = maximum_rotation(g_cur);

    for (int i = 0; i < 3 * nlocal; i++) {
      p_s[i] = -g_cur[i] * factor * scaling;
      g_old[i] = g_cur[i]  * factor;
      for (int k = 0; k < num_mem; k++){
        ds[k][i] = 0.0;
        dy[k][i] = 0.0;
      }
    }
    for (int k = 0; k < num_mem; k++)
      rho[k] = 0.0;

    } else {
    dyds = 0.0;
    for (int i = 0; i < 3 * nlocal; i++) {
      ds[m_index][i] = p_s[i];
      dy[m_index][i] = g_cur[i] - g_old[i];
      dyds += ds[m_index][i] * dy[m_index][i];
    }
    MPI_Allreduce(&dyds, &dyds_global, 1, MPI_DOUBLE, MPI_SUM, world);

    if (nreplica > 1) {
      dyds_global *= factor;
      dyds = dyds_global;
      MPI_Allreduce(&dyds, &dyds_global, 1,MPI_DOUBLE,MPI_SUM,universe->uworld);
    }

    if (fabs(dyds_global) > 1.0e-60) rho[m_index] = 1.0 / dyds_global;
    else rho[m_index] = 1.0e60;

    if (rho[m_index] < 0.0){
      local_iter = 0;
      return calc_search_direction();
    }

    for (int i = 0; i < 3 * nlocal; i++) {
      p_s[i] = g_cur[i];
    }

    // loop over last m indecies
    for(int k = num_mem - 1; k > -1; k--) {
      // this loop should run from the newest memory to the oldest one.

      c_ind = (k + m_index + 1) % num_mem;

      // dot product between dg and q

      sq = 0.0;
      for (int i = 0; i < 3 * nlocal; i++) {
        sq += ds[c_ind][i] * p_s[i];
      }
      MPI_Allreduce(&sq,&sq_global,1,MPI_DOUBLE,MPI_SUM,world);
      if (nreplica > 1) {
        sq_global *= factor;
        sq = sq_global;
        MPI_Allreduce(&sq,&sq_global,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      }

      // update alpha

      alpha[c_ind] = rho[c_ind] * sq_global;

      // update q

      for (int i = 0; i < 3 * nlocal; i++) {
        p_s[i] -= alpha[c_ind] * dy[c_ind][i];
      }
    }

    // dot product between dg with itself
    yy = 0.0;
    for (int i = 0; i < 3 * nlocal; i++) {
      yy += dy[m_index][i] * dy[m_index][i];
    }
    MPI_Allreduce(&yy,&yy_global,1,MPI_DOUBLE,MPI_SUM,world);
    if (nreplica > 1) {
      yy_global *= factor;
      yy = yy_global;
      MPI_Allreduce(&yy,&yy_global,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
    }

    // calculate now search direction

    double devis = rho[m_index] * yy_global;

    if (fabs(devis) > 1.0e-60) {
      for (int i = 0; i < 3 * nlocal; i++) {
        p_s[i] = factor * p_s[i] / devis;
      }
    }else{
      for (int i = 0; i < 3 * nlocal; i++) {
        p_s[i] = factor * p_s[i] * 1.0e60;
      }
    }

    for (int k = 0; k < num_mem; k++){
      // this loop should run from the oldest memory to the newest one.

      if (local_iter < num_mem) c_ind = k;
      else c_ind = (k + m_index + 1) % num_mem;

      // dot product between p and da
      yr = 0.0;
      for (int i = 0; i < 3 * nlocal; i++) {
        yr += dy[c_ind][i] * p_s[i];
      }

      MPI_Allreduce(&yr,&yr_global,1,MPI_DOUBLE,MPI_SUM,world);
      if (nreplica > 1) {
        yr_global *= factor;
        yr = yr_global;
        MPI_Allreduce(&yr,&yr_global,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      }

      beta = rho[c_ind] * yr_global;
      for (int i = 0; i < 3 * nlocal; i++) {
        p_s[i] += ds[c_ind][i] * (alpha[c_ind] - beta);
      }
    }
    if (use_line_search == 0)
      scaling = maximum_rotation(p_s);
    for (int i = 0; i < 3 * nlocal; i++) {
      p_s[i] = - factor * p_s[i] * scaling;
      g_old[i] = g_cur[i] * factor;
    }
  }
  local_iter++;
}

/* ----------------------------------------------------------------------
   rotation of spins along the search direction
---------------------------------------------------------------------- */

void MinSpinFP_LBFGS::advance_spins()
{
  int nlocal = atom->nlocal;
  double **sp = atom->sp;
  double normsp = 0.0;
  // loop on all spins on proc.
  for (int i = 0; i < nlocal; i++) {
    normsp = 0.0;
    for (int cc = 0; cc < 3; cc++) {
      sp[i][cc] += p_s[3 * i + cc];
      normsp += sp[i][cc] * sp[i][cc];
    }
    normsp = sqrt(normsp);
    for (int cc = 0; cc < 3; cc++)
      sp[i][cc] /= normsp;
  }
}

/* ----------------------------------------------------------------------
   See Jorge Nocedal and Stephen J. Wright 'Numerical
   Optimization' Second Edition, 2006 (p. 60)
---------------------------------------------------------------------- */

void MinSpinFP_LBFGS::make_step(double steplength)
{
  double p_scaled[3];
  int nlocal = atom->nlocal;
  double s_new[3];
  double **sp = atom->sp;
  double der_e_cur_tmp = 0.0;

  double normsp = 0.0;
  // loop on all spins on proc.
  for (int i = 0; i < nlocal; i++) {

    // scale the search direction

    for (int j = 0; j < 3; j++) p_scaled[j] = steplength * p_s[3 * i + j];

    normsp = 0.0;
    for (int cc = 0; cc < 3; cc++) {
      sp[i][cc] = sp_copy[i][cc] + p_scaled[cc];
      normsp += sp[i][cc] * sp[i][cc];
    }
    normsp = sqrt(normsp);
    for (int cc = 0; cc < 3; cc++)
      sp[i][cc] /= normsp;
  }

  ecurrent = energy_force(0);
  calc_gradient();
  neval++;
  der_e_cur = 0.0;
  for (int i = 0; i < 3 * nlocal; i++) {
    der_e_cur += g_cur[i] * p_s[i];
  }
  MPI_Allreduce(&der_e_cur,&der_e_cur_tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  der_e_cur = der_e_cur_tmp;
  if (update->multireplica == 1) {
    MPI_Allreduce(&der_e_cur_tmp,&der_e_cur,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
  }

  e_cur = ecurrent;
}

/* ---------------------------------------------------------------------
  Calculate step length which satisfies approximate or strong
  Wolfe conditions using the cubic interpolation
------------------------------------------------------------------------- */

int MinSpinFP_LBFGS::calc_and_make_step()
{
  double **sp = atom->sp;
  int nlocal = atom->nlocal;
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  int i = 1;
  int need_amax=1;

  double ei=e_cur;
  double dei=der_e_cur;
  double emax=0.0;
  double demax=0.0;

  double eim1=e_pr;
  double deim1=der_e_pr;

  double optstep=1.0;
  double prevstep=0.0;

  while (1){
    make_step(optstep);
    ei=e_cur;
    dei=der_e_cur;
    if (awc(der_e_pr, e_pr, dei, ei))
      break;

    tmp1 = e_pr + c1 * optstep * der_e_pr;
    if (ei > tmp1 || (ei >= eim1 && i > 1)){
      optstep = zoom(prevstep, optstep, eim1, deim1, ei, dei);
      break;
    }

    tmp1=fabs(dei);
    tmp2=-c2 * der_e_pr;
    if (tmp1 <= tmp2)
      break;
    if (dei >= 0.0){
      optstep = zoom(optstep, prevstep, ei, dei, eim1, deim1);
      break;
    }
    if (i == maxiterls)
      break;
    tmp1 = fabs(intervalsize - optstep);
    if (tmp1 < 1.0e-10){
      intervalsize = 1.5 * optstep;
      need_amax = 1;
    }
    // calculate things for maximum alpha
    if (need_amax){
      make_step(intervalsize);
      emax = e_cur;
      demax = der_e_cur;
      need_amax = 0;
      // check if max alpha satisfies awc:
      if (awc(der_e_pr, e_pr, emax, demax)){
        optstep = intervalsize;
        break;
      }
    }

    prevstep=optstep;
    optstep = cubic_interpolation(
            optstep, intervalsize,
            ei, dei, emax, demax);
    if (fabs(optstep-prevstep) < 1.0e-10){
      make_step(optstep);
      break;
    }
    eim1=ei;
    deim1=dei;
    i+=1;
  }

  MPI_Bcast(&optstep,1,MPI_DOUBLE,0,world);
  for (int i = 0; i < 3 * nlocal; i++) {
    p_s[i] = optstep * p_s[i];
  }
  return 0;
}

/* ----------------------------------------------------------------------
   See Jorge Nocedal and Stephen J. Wright 'Numerical
   Optimization' Second Edition, 2006 (p. 61)
---------------------------------------------------------------------- */

double MinSpinFP_LBFGS::zoom(double a_lo, double a_hi, double f_lo, double df_lo,
                           double f_hi, double df_hi){
  int i = 0;
  double ej;
  double dej;
  double tmp1;
  double a_j;

  while(1){
    a_j = cubic_interpolation(a_lo, a_hi, f_lo, f_hi, df_lo, df_hi);
    if (a_j < 1.0e-6){
      a_j = 1.0e-6;
      make_step(a_j);
      return a_j;
    }
    make_step(a_j);
    ej = e_cur;
    dej = der_e_cur;
    if (awc(der_e_pr, e_pr, dej, ej))
      return a_j;

    tmp1=e_pr + c1 * a_j * der_e_pr;
    if (ej > tmp1 || ej >= f_lo){
      a_hi = a_j;
      f_hi = ej;
      df_hi = dej;
    }
    else{
      if (fabs(dej) <= -c2*der_e_pr)
        return a_j;
      if (dej * (a_hi - a_lo) >= 0.0){
        a_hi = a_lo;
        f_hi = f_lo;
        df_hi = df_lo;
      }
      a_lo = a_j;
      f_lo = ej;
      df_lo = dej;
    }
    i+=1;

    if (fabs(a_lo - a_hi)< epsdx){
      // to small interval quit
      make_step(a_lo);
      return a_lo;
    }
    if (i==maxiterls){
      if (a_lo > epsdx){
        make_step(a_lo);
        return a_lo;
      } else{
        make_step(a_hi);
        return a_hi;
      }
    }
  }
}

/* ----------------------------------------------------------------------
  Approximate Wolfe Conditions
  Hager W.M. and H. Zhang, SIAM J. Optim. Vol. 16, No. 1, pp 170-192
------------------------------------------------------------------------- */

int MinSpinFP_LBFGS::awc(double der_phi_0, double phi_0,
                      double der_phi_j, double phi_j){

  double eps = 1.0e-6;
  double delta = 0.1;
  double sigma = 0.9;
  double dsnt =  phi_0 + eps * fabs(phi_0);
  double tmp1 = (2.0*delta - 1.0) * der_phi_0;
  double tmp2 = sigma * der_phi_0;

  if (phi_j <= dsnt && tmp1 >= der_phi_j && der_phi_j>= tmp2)
    return 1;
  else
    return 0;
}

double MinSpinFP_LBFGS::maximum_rotation(double *p)
{
  double norm2,norm2_global,scaling,gamma;
  int nlocal = atom->nlocal;
  int ntotal = 0;

  norm2 = 0.0;
  for (int i = 0; i < 3 * nlocal; i++) norm2 += p[i] * p[i];

  MPI_Allreduce(&norm2,&norm2_global,1,MPI_DOUBLE,MPI_SUM,world);
  if (nreplica > 1) {
    norm2 = norm2_global;
    MPI_Allreduce(&norm2,&norm2_global,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
  }
  MPI_Allreduce(&nlocal,&ntotal,1,MPI_INT,MPI_SUM,world);
  if (nreplica > 1) {
    nlocal = ntotal;
    MPI_Allreduce(&nlocal,&ntotal,1,MPI_INT,MPI_SUM,universe->uworld);
  }

  scaling = (maxepsrot * sqrt((double) ntotal / norm2_global));

  if (scaling < 1.0) gamma = scaling;
  else gamma = 1.0;

  return gamma;
}

double MinSpinFP_LBFGS::cubic_interpolation(
        double x_0, double x_1, double f_0, double f_1,
        double df_0, double df_1){

  if (x_0 > x_1){
    double tmp=0.0;
    tmp = x_0;
    x_0 = x_1;
    x_1 = tmp;
    tmp = f_0;
    f_0 = f_1;
    f_1 = tmp;
    tmp = df_0;
    df_0 = df_1;
    df_1 = tmp;
  }

  double r = x_1 - x_0;
  double a = -2.0 * (f_1 - f_0)/(r*r*r) + (df_1 + df_0)/(r*r);
  double b = 3.0 * (f_1 - f_0)/(r*r)-(df_1 + 2.0 * df_0)/r;
  double c = df_0;
  double d = f_0;
  double D = b * b - 3.0 * a * c;
  double r0;
  double f_r0;
  double deltax;
  double minstep=0.0;

  if (D < 0.0){
    if (f_0 < f_1) minstep = x_0;
    else minstep = x_1;
  }
  else {
    r0 = (-b + sqrt(D)) / (3.0 * a) + x_0;
    if (x_0 < r0 && r0 < x_1){
      deltax=r0 - x_0;
      f_r0 = a*(deltax*deltax*deltax) + b*(deltax*deltax) + c*deltax + d;
      if (f_0 > f_r0 and f_1 > f_r0)
        minstep = r0;
      else{
        if (f_0 < f_1) minstep = x_0;
        else minstep = x_1;
      }
    }
    else{
      if (f_0 < f_1) minstep = x_0;
      else minstep = x_1;
    }
  }
  return minstep;
}