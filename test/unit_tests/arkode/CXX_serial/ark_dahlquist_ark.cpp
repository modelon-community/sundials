/* ---------------------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 * ---------------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2022, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * ---------------------------------------------------------------------------
 * IMEX multirate Dahlquist problem:
 *
 * y' = lambda_e * y + lambda_i * y + lambda_f * y
 * ---------------------------------------------------------------------------*/

// Header files
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <cmath>

#include <arkode/arkode_arkstep.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

// Constants
#define NEG_ONE SUN_RCONST(-1.0)
#define ZERO    SUN_RCONST(0.0)
#define ONE     SUN_RCONST(1.0)

// Method types
enum class method_type { expl, impl, imex };

// User data structure
struct UserData
{
  sunrealtype lambda_e = NEG_ONE;
  sunrealtype lambda_i = NEG_ONE;
};

// User-supplied Functions called by the solver
int fe(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);
int fi(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);
int Ji(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
       void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

// Private function to check function return values
int check_flag(void *flagvalue, const std::string funcname, int opt);

// Test drivers
int run_tests(method_type type, sunrealtype t0, int nsteps, sunrealtype h,
              sunrealtype reltol, sunrealtype abstol, UserData* udata,
              SUNContext ctx);


// -----------------------------------------------------------------------------
// Main Program
// -----------------------------------------------------------------------------


int main(int argc, char* argv[])
{
  // Initial time
  sunrealtype t0 = ZERO;

  // Number of time steps
  int nsteps = 1;

  // Relative and absolute tolerances
  sunrealtype reltol = SUN_RCONST(1.0e-4);
  sunrealtype abstol = SUN_RCONST(1.0e-6);

  // Step size
  sunrealtype h = SUN_RCONST(0.01);

  // User data structure
  UserData udata;

  // Check for inputs
  if (argc > 1) udata.lambda_e = std::stod(argv[1]);
  if (argc > 2) udata.lambda_i = std::stod(argv[2]);
  if (argc > 3) h = std::stod(argv[3]);
  if (argc > 4) nsteps = std::stoi(argv[4]);

  // Output problem setup
  std::cout  << "\nDahlquist ODE test problem:\n"
             << "  lambda expl  = " << udata.lambda_e << "\n"
             << "  lambda impl  = " << udata.lambda_i << "\n"
             << "  step size    = " << h << "\n"
             << "  relative tol = " << reltol << "\n"
             << "  absolute tol = " << abstol << "\n";

  // Create SUNDIALS context
  sundials::Context sunctx;

  // Test methods
  int numfails = 0;

  // Explicit
  numfails += run_tests(method_type::expl, t0, nsteps, h, reltol, abstol,
                        &udata, sunctx);

  // Implicit
  numfails += run_tests(method_type::impl, t0, nsteps, h, reltol, abstol,
                        &udata, sunctx);

  // IMEX
  numfails += run_tests(method_type::imex, t0, nsteps, h, reltol, abstol,
                        &udata, sunctx);

  if (numfails)
  {
    std::cout << "\n\nFailed " << numfails << " tests!\n";
  }
  else
  {
    std::cout << "\n\nAll tests passed!\n";
  }

  // Return test status
  return numfails;
}


// -----------------------------------------------------------------------------
// Test drivers
// -----------------------------------------------------------------------------


int run_tests(method_type type, sunrealtype t0, int nsteps, sunrealtype h,
              sunrealtype reltol, sunrealtype abstol, UserData* udata,
              SUNContext sunctx)
{
  // Reusable error-checking flag
  int flag;

  // Test failure counter
  int numfails = 0;

  // Create initial condition vector
  N_Vector y = N_VNew_Serial(1, sunctx);
  if (check_flag((void *)y, "N_VNew_Serial", 0)) return 1;

  N_VConst(SUN_RCONST(1.0), y);

  // Create matrix and linear solver (if necessary)
  SUNMatrix       A  = nullptr;
  SUNLinearSolver LS = nullptr;

  if (type == method_type::impl || type == method_type::imex)
  {
    // Initialize dense matrix data structures and solvers
    A = SUNDenseMatrix(1, 1, sunctx);
    if (check_flag((void *)A, "SUNDenseMatrix", 0)) return 1;

    LS = SUNLinSol_Dense(y, A, sunctx);
    if (check_flag((void *)LS, "SUNLinSol_Dense", 0)) return 1;
  }

  // -----------------
  // Create integrator
  // -----------------

  // Create integrator based on type
  void* arkstep_mem = nullptr;

  if (type == method_type::expl)
  {
    arkstep_mem = ARKStepCreate(fe, nullptr, t0, y, sunctx);
  }
  else if (type == method_type::impl)
  {
    arkstep_mem = ARKStepCreate(nullptr, fi, t0, y, sunctx);
  }
  else if (type == method_type::imex)
  {
    arkstep_mem = ARKStepCreate(fe, fi, t0, y, sunctx);
  }
  else
  {
    return 1;
  }
  if (check_flag((void *) arkstep_mem, "ARKStepCreate", 0)) return 1;

  // Set user data
  flag = ARKStepSetUserData(arkstep_mem, udata);
  if (check_flag(&flag, "ARKStepSetUserData", 1)) return 1;

  // Specify tolerances
  flag = ARKStepSStolerances(arkstep_mem, reltol, abstol);
  if (check_flag(&flag, "ARKStepSStolerances", 1)) return 1;

  // Specify fixed time step size
  flag = ARKStepSetFixedStep(arkstep_mem, h);
  if (check_flag(&flag, "ARKStepSetFixedStep", 1)) return 1;

  if (type == method_type::impl || type == method_type::imex)
  {
    // Attach linear solver
    flag = ARKStepSetLinearSolver(arkstep_mem, LS, A);
    if (check_flag(&flag, "ARKStepSetLinearSolver", 1)) return 1;

    // Set Jacobian function
    flag = ARKStepSetJacFn(arkstep_mem, Ji);
    if (check_flag(&flag, "ARKStepSetJacFn", 1)) return 1;

    // Specify linearly implicit RHS, with non-time-dependent Jacobian
    flag = ARKStepSetLinear(arkstep_mem, 0);
    if (check_flag(&flag, "ARKStepSetLinear", 1)) return 1;
  }

  // ---------------------------
  // Evolve with various methods
  // ---------------------------

  // Methods to test
  int num_methods;

  if (type == method_type::expl)
  {
    std::cout << "\n========================\n"
              << "Test explicit RK methods\n"
              << "========================\n";
    num_methods = 1;
  }
  else if (type == method_type::impl)
  {
    std::cout << "\n========================\n"
              << "Test implicit RK methods\n"
              << "========================\n";
    num_methods = 1;
  }
  else if (type == method_type::imex)
  {
    std::cout << "\n=====================\n"
              << "Test IMEX ARK methods\n"
              << "=====================\n";
    num_methods = 1;
  }
  else
  {
    return 1;
  }

  for (int i = 0; i < num_methods; i++)
  {
    std::cout << "\nTesting method " << i << "\n";

    // -------------
    // Select method
    // -------------

    ARKodeButcherTable Be = nullptr;
    ARKodeButcherTable Bi = nullptr;

    if (type == method_type::expl)
    {
      // Explicit Euler
      Be = ARKodeButcherTable_Alloc(1, SUNFALSE);
      Be->A[0][0] = ZERO;
      Be->b[0] = ONE;
      Be->c[0] = ZERO;
      Be->q = 1;
      Bi = nullptr;
    }
    else if (type == method_type::impl)
    {
      // Implicit Euler
      Bi = ARKodeButcherTable_Alloc(1, SUNFALSE);
      Bi->A[0][0] = ONE;
      Bi->b[0] = ONE;
      Bi->c[0] = ONE;
      Bi->q = 1;
      Be = nullptr;
    }
    else if (type == method_type::imex)
    {
      // IMEX Euler
      Be = ARKodeButcherTable_Alloc(2, SUNFALSE);
      Be->A[1][0] = ONE;
      Be->b[0] = ONE;
      Be->c[1] = ONE;
      Be->q = 1;

      Bi = ARKodeButcherTable_Alloc(2, SUNFALSE);
      Bi->A[1][1] = ONE;
      Bi->b[1] = ONE;
      Bi->c[1] = ONE;
      Bi->q = 1;
    }
    else
    {
      return 1;
    }

    // Attach Butcher tables
    flag = ARKStepSetTables(arkstep_mem, 1, 0, Bi, Be);
    if (check_flag(&flag, "ARKStepSetTables", 1)) return 1;

    ARKodeButcherTable_Free(Be);
    ARKodeButcherTable_Free(Bi);
    Be = nullptr;
    Bi = nullptr;

    // --------------
    // Evolve in time
    // --------------

    sunrealtype t  = t0;
    sunrealtype tf = nsteps * h;

    for (int i = 0; i < nsteps; i++)
    {
      // Advance in time
      flag = ARKStepEvolve(arkstep_mem, tf, y, &t, ARK_ONE_STEP);
      if (check_flag(&flag, "ARKStepEvolve", 1)) return 1;

      // Update output time
      tf += h;
    }

    // -----------------
    // Output statistics
    // -----------------

    long int nst, nfe, nfi;        // integrator
    long int nni, ncfn;            // nonlinear solver
    long int nsetups, nje, nfeLS;  // linear solver

    flag = ARKStepGetNumSteps(arkstep_mem, &nst);
    if (check_flag(&flag, "ARKStepGetNumSteps", 1)) return 1;

    flag = ARKStepGetNumRhsEvals(arkstep_mem, &nfe, &nfi);
    if (check_flag(&flag, "ARKStepGetNumRhsEvals", 1)) return 1;

    if (type == method_type::impl || type == method_type::imex)
    {
      flag = ARKStepGetNumNonlinSolvIters(arkstep_mem, &nni);
      if (check_flag(&flag, "ARKStepGetNumNonlinSolvIters", 1)) return 1;

      flag = ARKStepGetNumNonlinSolvConvFails(arkstep_mem, &ncfn);
      if (check_flag(&flag, "ARKStepGetNumNonlinSolvConvFails", 1)) return 1;

      flag = ARKStepGetNumLinSolvSetups(arkstep_mem, &nsetups);
      if (check_flag(&flag, "ARKStepGetNumLinSolvSetups", 1)) return 1;

      flag = ARKStepGetNumJacEvals(arkstep_mem, &nje);
      if (check_flag(&flag, "ARKStepGetNumJacEvals", 1)) return 1;

      flag = ARKStepGetNumLinRhsEvals(arkstep_mem, &nfeLS);
      check_flag(&flag, "ARKStepGetNumLinRhsEvals", 1);
    }

    sunrealtype pow = ZERO;
    if (type == method_type::expl || type == method_type::imex)
    {
      pow += udata->lambda_e;
    }
    if (type == method_type::impl || type == method_type::imex)
    {
      pow += udata->lambda_i;
    }
    sunrealtype ytrue = exp(pow * t);

    sunrealtype* ydata = N_VGetArrayPointer(y);
    sunrealtype  error = ytrue - ydata[0];

    std::cout << "\nARKStep Statistics:\n"
              << "  Time        = " << t        << "\n"
              << "  y(t)        = " << ytrue    << "\n"
              << "  y_n         = " << ydata[0] << "\n"
              << "  Error       = " << error    << "\n"
              << "  Steps       = " << nst      << "\n"
              << "  Fe evals    = " << nfe      << "\n"
              << "  Fi evals    = " << nfi      << "\n";

    if (type == method_type::impl || type == method_type::imex)
    {
      std::cout << "  NLS iters   = " << nni     << "\n"
                << "  NLS fails   = " << ncfn    << "\n"
                << "  LS setups   = " << nsetups << "\n"
                << "  LS Fi evals = " << nfeLS   << "\n"
                << "  Ji evals    = " << nje     << "\n";
    }

    // ----------------
    // Check statistics
    // ----------------

    // expected number of explicit functions evaluations
    if (type == method_type::expl || type == method_type::imex)
    {
      if (nfe != nst + 1)
      {
        numfails++;
        std::cout << "Fe RHS evals:\n"
                  << "  actual:   " << nfe << "\n"
                  << "  expected: " << nst + 1 << "\n";
      }
    }

    // expected number of implicit functions evaluations
    if (type == method_type::impl || type == method_type::imex)
    {
      if (nfi != nst + 1 + nni)
      {
        numfails++;
        std::cout << "Fi RHS evals:\n"
                  << "  actual:   " << nfi << "\n"
                  << "  expected: " << nst + nni << "\n";
      }
    }

    if (numfails)
    {
      std::cout << "Failed " << numfails << " checks\n";
    }
    else
    {
      std::cout << "All checks passed\n";
    }

    // -------------------
    // Setup for next test
    // -------------------

    // Free table(s)

    // Reset state vector to the initial condition
    N_VConst(SUN_RCONST(1.0), y);

    // Re-initialize integrator based on type
    if (type == method_type::expl)
    {
      flag = ARKStepReInit(arkstep_mem, fe, nullptr, t0, y);
    }
    else if (type == method_type::impl)
    {
      flag = ARKStepReInit(arkstep_mem, nullptr, fi, t0, y);
    }
    else if (type == method_type::imex)
    {
      flag = ARKStepReInit(arkstep_mem, fe, fi, t0, y);
    }
    else
    {
      return 1;
    }
    if (check_flag(&flag, "ARKStepReInit", 1)) return 1;
  }

  // Clean up
  ARKStepFree(&arkstep_mem);
  if (type == method_type::impl || type == method_type::imex)
  {
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
  }
  N_VDestroy(y);

  return numfails;
}


// -----------------------------------------------------------------------------
// Functions called by the solver
// -----------------------------------------------------------------------------


// Explicit ODE RHS function fe(t,y)
int fe(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  sunrealtype* y_data  = N_VGetArrayPointer(y);
  sunrealtype* yd_data = N_VGetArrayPointer(ydot);
  UserData* udata      = static_cast<UserData*>(user_data);

  yd_data[0] = udata->lambda_e * y_data[0];

  return 0;
}

// Implicit ODE RHS function fi(t,y)
int fi(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  sunrealtype* y_data  = N_VGetArrayPointer(y);
  sunrealtype* yd_data = N_VGetArrayPointer(ydot);
  UserData* udata      = static_cast<UserData*>(user_data);

  yd_data[0] = udata->lambda_i * y_data[0];

  return 0;
}

// Jacobian routine to compute J(t,y) = dfi/dy.
int Ji(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
       N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  sunrealtype* J_data = SUNDenseMatrix_Data(J);
  UserData* udata     = static_cast<UserData*>(user_data);

  J_data[0] = udata->lambda_i;

  return 0;
}


// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------


// Check function return value
int check_flag(void *flagvalue, const std::string funcname, int opt)
{
  int *errflag;

  // Check if function returned NULL pointer - no memory allocated
  if (opt == 0 && flagvalue == nullptr)
  {

    std::cerr << "\nMEMORY_ERROR: " << funcname << " failed - returned NULL pointer\n\n";
    return 1;
  }
  // Check if flag < 0
  else if (opt == 1)
  {
    errflag = (int *) flagvalue;
    if (*errflag < 0)
    {
      std::cerr << "\nSUNDIALS_ERROR: " << funcname << " failed with flag = " << *errflag << "\n\n";
      return 1;
    }
  }

  return 0;
}
