#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <stdlib.h> // for malloc, free
#include <string.h> // for memcpy

// Error checking macro
#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",      \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                         \
    }                                                    \
} while (0)

// Helper macro for 3D array indexing from a 1D pointer (Row-Major: i, j, k)
#define IDX(i, j, k, Ny, Nz) ((i) * (Ny) * (Nz) + (j) * (Nz) + (k))

// Helper function to flatten the 3D host array (FPfield***) into a 1D contiguous array (FPfield*)
// This is necessary because C++ 3D array allocation (array of pointers to pointers) is not contiguous.
FPfield* flatten_3D_array(FPfield*** h_array, int nx, int ny, int nz) {
    // Total size in bytes
    size_t size = nx * ny * nz * sizeof(FPfield);
    FPfield* flat_array = (FPfield*)malloc(size);
    if (flat_array == NULL) return NULL;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            // Copy the contiguous block of 'nz' elements for the current i and j index
            // Destination pointer: flat_array + offset
            // The offset is calculated by the 1D index formula
            memcpy(flat_array + IDX(i, j, 0, ny, nz), h_array[i][j], nz * sizeof(FPfield));
        }
    }
    return flat_array;
}

// -------------------------------------------------------------------------------- //

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

// --------------- we use the GPU mover! -----------------

// GPU kernel for particle mover
// Each thread processes ONE particle through ALL subcycles
__global__ void mover_PC_kernel(
    // particles
    FPpart* x, FPpart* y, FPpart* z,    // space
    FPpart* u, FPpart* v, FPpart* w,    // velocity
    // fields (1D POINTERS)
    FPfield* d_Ex_flat, FPfield* d_Ey_flat, FPfield* d_Ez_flat,
    FPfield* d_Bxn_flat, FPfield* d_Byn_flat, FPfield* d_Bzn_flat,
    // space grid (1D POINTERS)
    FPfield* d_XN_flat, FPfield* d_YN_flat, FPfield* d_ZN_flat,
    // params for the grid
    FPfield xStart, FPfield yStart, FPfield zStart,
    FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
    FPfield Lx, FPfield Ly, FPfield Lz,
    // Grid dimensions for 1D indexing
    int nxn, int nyn, int nzn,
    // params for simulation
    FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2,
    int n_sub_cycles, int NiterMover,
    bool PERIODICX, bool PERIODICY, bool PERIODICZ,
    // number of particles (!)
    int nop)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // we divide the particles
    
    // check if thread index is within bounds
    if (i >= nop) return;
    
    // Local variables for this particle
    FPfield Exl, Eyl, Ezl, Bxl, Byl, Bzl;
    int ix, iy, iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // start subcycling (sequential subcycling for each particle)
    for (int i_sub = 0; i_sub < n_sub_cycles; i_sub++) {
        
        // initial position for this subcycle
        xptilde = x[i];
        yptilde = y[i];
        zptilde = z[i];
        
        // calculate the average velocity iteratively
        // predictor-corrector iterations
        for (int innter = 0; innter < NiterMover; innter++) {
            
            // interpolation G-->P
            ix = 2 + int((x[i] - xStart) * invdx);
            iy = 2 + int((y[i] - yStart) * invdy);
            iz = 2 + int((z[i] - zStart) * invdz);
            
            // calculate weights
            // Use IDX macro for 1D access to grid nodes
            xi[0]   = x[i] - d_XN_flat[IDX(ix - 1, iy, iz, nyn, nzn)];
            eta[0]  = y[i] - d_YN_flat[IDX(ix, iy - 1, iz, nyn, nzn)];
            zeta[0] = z[i] - d_ZN_flat[IDX(ix, iy, iz - 1, nyn, nzn)];
            xi[1]   = d_XN_flat[IDX(ix, iy, iz, nyn, nzn)] - x[i];
            eta[1]  = d_YN_flat[IDX(ix, iy, iz, nyn, nzn)] - y[i];
            zeta[1] = d_ZN_flat[IDX(ix, iy, iz, nyn, nzn)] - z[i];
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * invVOL;
            
            // set to zero local electric and magnetic field
            Exl = 0.0; Eyl = 0.0; Ezl = 0.0;
            Bxl = 0.0; Byl = 0.0; Bzl = 0.0;
            
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        // Use IDX macro for 1D access to fields
                        int field_idx = IDX(ix - ii, iy - jj, iz - kk, nyn, nzn);
                        Exl += weight[ii][jj][kk] * d_Ex_flat[field_idx];
                        Eyl += weight[ii][jj][kk] * d_Ey_flat[field_idx];
                        Ezl += weight[ii][jj][kk] * d_Ez_flat[field_idx];
                        Bxl += weight[ii][jj][kk] * d_Bxn_flat[field_idx];
                        Byl += weight[ii][jj][kk] * d_Byn_flat[field_idx];
                        Bzl += weight[ii][jj][kk] * d_Bzn_flat[field_idx];
                    }
            // end interpolation
            
            // Boris particle pusher
            omdtsq = qomdt2 * qomdt2 * (Bxl*Bxl + Byl*Byl + Bzl*Bzl);
            denom = 1.0 / (1.0 + omdtsq);
            // solve the position equation
            ut = u[i] + qomdt2 * Exl;
            vt = v[i] + qomdt2 * Eyl;
            wt = w[i] + qomdt2 * Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut + qomdt2*(vt*Bzl - wt*Byl + qomdt2*udotb*Bxl)) * denom;
            vptilde = (vt + qomdt2*(wt*Bxl - ut*Bzl + qomdt2*udotb*Byl)) * denom;
            wptilde = (wt + qomdt2*(ut*Byl - vt*Bxl + qomdt2*udotb*Bzl)) * denom;
            // update position (half-step)
            x[i] = xptilde + uptilde * dto2;
            y[i] = yptilde + vptilde * dto2;
            z[i] = zptilde + wptilde * dto2;
            
        } // end iterator loop
        
        // update the final position and velocity
        u[i] = 2.0 * uptilde - u[i];
        v[i] = 2.0 * vptilde - v[i];
        w[i] = 2.0 * wptilde - w[i];
        x[i] = xptilde + uptilde * dt_sub_cycling;
        y[i] = yptilde + vptilde * dt_sub_cycling;
        z[i] = zptilde + wptilde * dt_sub_cycling;
        
        //////////
        //////////
        ////////// BC

        // X-DIRECTION: BC particles
        if (x[i] > Lx) {
            if (PERIODICX) { // PERIODIC
                x[i] = x[i] - Lx;
            } else { // REFLECTING BC
                u[i] = -u[i];
                x[i] = 2.0 * Lx - x[i];
            }
        }
        if (x[i] < 0) {
            if (PERIODICX) { // PERIODIC
                x[i] = x[i] + Lx;
            } else { // REFLECTING BC
                u[i] = -u[i];
                x[i] = -x[i];
            }
        }
        
        // Y-DIRECTION: BC particles
        if (y[i] > Ly) {
            if (PERIODICY) { // PERIODIC
                y[i] = y[i] - Ly;
            } else { // REFLECTING BC
                v[i] = -v[i];
                y[i] = 2.0 * Ly - y[i];
            }
        }
        if (y[i] < 0) {
            if (PERIODICY) { // PERIODIC
                y[i] = y[i] + Ly;
            } else { // REFLECTING BC
                v[i] = -v[i];
                y[i] = -y[i];
            }
        }
        
        // Z-DIRECTION: BC particles
        if (z[i] > Lz) {
            if (PERIODICZ) { // PERIODIC
                z[i] = z[i] - Lz;
            } else { // REFLECTING BC
                w[i] = -w[i];
                z[i] = 2.0 * Lz - z[i];
            }
        }
        if (z[i] < 0) {
            if (PERIODICZ) { // PERIODIC
                z[i] = z[i] + Lz;
            } else { // REFLECTING BC
                w[i] = -w[i];
                z[i] = -z[i];
            }
        }
        
    } // end subcycling loop
} // end of the mover

/** GPU version of particle mover */
int mover_PC_gpu(struct particles* part, struct EMfield* field, 
                 struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "*** GPU MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt / ((double) part->n_sub_cycles);
    FPpart dto2 = 0.5 * dt_sub_cycling;
    FPpart qomdt2 = part->qom * dto2 / param->c;
    
    int nop = part->nop;
    
    // Grid dimensions (ASSUMPTION: nxn, nyn, nzn exist in struct grid)
    int nxn = grd->nxn; 
    int nyn = grd->nyn;
    int nzn = grd->nzn;
    size_t field_size = nxn * nyn * nzn * sizeof(FPfield); // Total size of 1D field array
    
    
    // @@ 1. PARTICLE DATA ALLOCATION & COPY
    
    // allocate device memory for particles
    FPpart *d_x, *d_y, *d_z, *d_u, *d_v, *d_w;
    size_t particle_size = nop * sizeof(FPpart);
    
    CHECK(cudaMalloc(&d_x, particle_size));
    CHECK(cudaMalloc(&d_y, particle_size));
    CHECK(cudaMalloc(&d_z, particle_size));
    CHECK(cudaMalloc(&d_u, particle_size));
    CHECK(cudaMalloc(&d_v, particle_size));
    CHECK(cudaMalloc(&d_w, particle_size));
    
    // copy particle data to device
    CHECK(cudaMemcpy(d_x, part->x, particle_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, part->y, particle_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_z, part->z, particle_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_u, part->u, particle_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, part->v, particle_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, part->w, particle_size, cudaMemcpyHostToDevice));
    
    
    // @@ 2. FIELD/GRID DATA ALLOCATION & COPY (FLATTENING 3D ARRAYS)

    // Pointers for 1D contiguous device arrays
    FPfield *d_Ex_flat, *d_Ey_flat, *d_Ez_flat;
    FPfield *d_Bxn_flat, *d_Byn_flat, *d_Bzn_flat;
    FPfield *d_XN_flat, *d_YN_flat, *d_ZN_flat;
    
    // Pointers for 1D contiguous host arrays (temporary for copying)
    FPfield *h_Ex_flat, *h_Ey_flat, *h_Ez_flat;
    FPfield *h_Bxn_flat, *h_Byn_flat, *h_Bzn_flat;
    FPfield *h_XN_flat, *h_YN_flat, *h_ZN_flat;

    // Allocate 1D device memory for fields/grid nodes
    CHECK(cudaMalloc(&d_Ex_flat, field_size));
    CHECK(cudaMalloc(&d_Ey_flat, field_size));
    CHECK(cudaMalloc(&d_Ez_flat, field_size));
    CHECK(cudaMalloc(&d_Bxn_flat, field_size));
    CHECK(cudaMalloc(&d_Byn_flat, field_size));
    CHECK(cudaMalloc(&d_Bzn_flat, field_size));
    CHECK(cudaMalloc(&d_XN_flat, field_size));
    CHECK(cudaMalloc(&d_YN_flat, field_size));
    CHECK(cudaMalloc(&d_ZN_flat, field_size));

    // Flatten host 3D arrays to 1D contiguous host arrays
    h_Ex_flat = flatten_3D_array(field->Ex, nxn, nyn, nzn);
    h_Ey_flat = flatten_3D_array(field->Ey, nxn, nyn, nzn);
    h_Ez_flat = flatten_3D_array(field->Ez, nxn, nyn, nzn);
    h_Bxn_flat = flatten_3D_array(field->Bxn, nxn, nyn, nzn);
    h_Byn_flat = flatten_3D_array(field->Byn, nxn, nyn, nzn);
    h_Bzn_flat = flatten_3D_array(field->Bzn, nxn, nyn, nzn);
    h_XN_flat = flatten_3D_array(grd->XN, nxn, nyn, nzn);
    h_YN_flat = flatten_3D_array(grd->YN, nxn, nyn, nzn);
    h_ZN_flat = flatten_3D_array(grd->ZN, nxn, nyn, nzn);
    
    // Copy 1D host field arrays to 1D device arrays
    CHECK(cudaMemcpy(d_Ex_flat, h_Ex_flat, field_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Ey_flat, h_Ey_flat, field_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Ez_flat, h_Ez_flat, field_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Bxn_flat, h_Bxn_flat, field_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Byn_flat, h_Byn_flat, field_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Bzn_flat, h_Bzn_flat, field_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_XN_flat, h_XN_flat, field_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_YN_flat, h_YN_flat, field_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ZN_flat, h_ZN_flat, field_size, cudaMemcpyHostToDevice));

    // Free temporary host flattened arrays
    free(h_Ex_flat); free(h_Ey_flat); free(h_Ez_flat);
    free(h_Bxn_flat); free(h_Byn_flat); free(h_Bzn_flat);
    free(h_XN_flat); free(h_YN_flat); free(h_ZN_flat);
    
    // @@ 3. KERNEL LAUNCH
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (nop + threadsPerBlock - 1) / threadsPerBlock;
    
    mover_PC_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_z, d_u, d_v, d_w,
        // Pass 1D device field/grid pointers
        d_Ex_flat, d_Ey_flat, d_Ez_flat, d_Bxn_flat, d_Byn_flat, d_Bzn_flat,
        d_XN_flat, d_YN_flat, d_ZN_flat,
        grd->xStart, grd->yStart, grd->zStart,
        grd->invdx, grd->invdy, grd->invdz, grd->invVOL,
        grd->Lx, grd->Ly, grd->Lz,
        // Pass grid dimensions
        nxn, nyn, nzn,
        dt_sub_cycling, dto2, qomdt2,
        part->n_sub_cycles, part->NiterMover,
        param->PERIODICX, param->PERIODICY, param->PERIODICZ,
        nop
    );
    
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    // @@ 4. COPY BACK & FREE
    
    // copy back particle data to the host
    CHECK(cudaMemcpy(part->x, d_x, particle_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(part->y, d_y, particle_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(part->z, d_z, particle_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(part->u, d_u, particle_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(part->v, d_v, particle_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(part->w, d_w, particle_size, cudaMemcpyDeviceToHost));
    
    // free device memory (particles)
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    CHECK(cudaFree(d_u));
    CHECK(cudaFree(d_v));
    CHECK(cudaFree(d_w));
    
    // free device memory (fields/grid)
    CHECK(cudaFree(d_Ex_flat));
    CHECK(cudaFree(d_Ey_flat));
    CHECK(cudaFree(d_Ez_flat));
    CHECK(cudaFree(d_Bxn_flat));
    CHECK(cudaFree(d_Byn_flat));
    CHECK(cudaFree(d_Bzn_flat));
    CHECK(cudaFree(d_XN_flat));
    CHECK(cudaFree(d_YN_flat));
    CHECK(cudaFree(d_ZN_flat));
    
    return 0;
}

// @@ InterpP2G (same as previous version)

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}