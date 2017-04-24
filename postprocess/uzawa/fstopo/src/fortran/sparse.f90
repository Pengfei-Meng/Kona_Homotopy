! The following code implements sparse matrix methods for use with
! stress-constrained topology optimization. These subroutines are
! called from python once they are wrapped with f2py.
!
! Copyright (c) Graeme J. Kennedy 2015, All rights reserved.

subroutine computeCSRToCSC(n, m, ncols, rowp, cols, colp, rows)
  ! Take the input non-zero data in the CSR format and create an
  ! equivalent CSC format for output.
  !
  ! Input:
  ! n:     the number of rows in the matrix
  ! m:     the number of columns in the matrix
  ! ncols: the number of non-zero entries
  ! rowp:  the row pointer in the matrix
  ! cols:  the column indices
  !
  ! Output:
  ! colp:  pointer to the columns for the CSC data
  ! rows:  the row indicies for the CSC data

  implicit none

  ! The input/ouput data
  integer, intent(in) :: n, m, ncols, rowp(n+1), cols(ncols)
  integer, intent(inout) :: colp(m+1), rows(ncols)

  ! The internal data
  integer :: i, j, jp

  ! Zero the matrix data
  colp(:) = 0
  rows(:) = 0

  do i = 1, n
     ! Count up the non-zero contributions to each column
     do jp = rowp(i), rowp(i+1)-1
        j = cols(jp)
        colp(j+1) = colp(j+1) + 1
     end do
  end do

  ! Modify the colp array so that it points into prows
  colp(1) = 1
  do i = 1, m
     colp(i+1) = colp(i) + colp(i+1)
  end do

  ! Next, scan through the matrix P and generate the CSC data
  do i = 1, n
     do jp = rowp(i), rowp(i+1)-1
        j = cols(jp)
        rows(colp(j)) = i
        colp(j) = colp(j) + 1
     end do
  end do

  ! Reset the pointer array
  do i = m, 1, -1
     colp(i+1) = colp(i)
  end do
  colp(1) = 1

end subroutine computeCSRToCSC

subroutine computeKmatNzPattern(n, ne, conn, rowp, ncols, cols, info)
  ! Compute the non-zero pattern of the stiffness matrix given the
  ! connectivity
  !
  ! Input:
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! conn:     the element connectivity
  !
  ! Output:
  ! rowp:     the row pointer
  ! ncols:    the number of columns
  ! cols:     the column index
  ! info:     successful = 0, otherwise the required length of ncols

  use precision
  use quicksort
  implicit none

  ! The input data
  integer, intent(in) :: n, ne, ncols, conn(4, ne)
  integer, intent(inout) :: rowp(n+1), cols(ncols)
  integer, intent(out) :: info

  ! Store an array of the non-zero entries
  integer :: i, j, k
  integer :: rp, rstart, rend, index, nzeros(n)

  ! All entries in the row pointer
  rowp(:) = 0

  ! Compute the maximum number of entries that we'll put in each row
  do i = 1, ne
     do j = 1, 4
        ! Count up the number of entries in the matrix
        rowp(conn(j, i)+1) = rowp(conn(j, i)+1) + 4
     end do
  end do

  ! Count it up so that we'll have enough room
  rowp(1) = 1
  do i = 1, n
     rowp(i+1) = rowp(i) + rowp(i+1)
  end do

  ! Return that we have failed, and we need a larger array
  if (ncols < rowp(n+1)-1) then
     info = rowp(n+1)-1
     return
  end if

  ! We have enough room to store the whole array
  info = 0

  ! Add the non-zero pattern from each element
  do i = 1, ne
     ! Add the non-zero pattern from the element into the matrix
     do j = 1, 4
        rp = rowp(conn(j, i))
        do k = 1, 4
           cols(rp) = conn(k, i)
           rp = rp + 1
        end do
        rowp(conn(j, i)) = rp
     end do
  end do

  ! Reset the pointer array
  do i = n, 1, -1
     rowp(i+1) = rowp(i)
  end do
  rowp(1) = 1

  ! Now, we've over-counted the entries, remove duplicates in each
  ! row. Note that this is an O(n) run time, but also has an O(n)
  ! storage requirement.
  nzeros(:) = 0

  index = 1
  rstart = 1
  do i = 1, n
     rend = rowp(i+1)

     ! Overwrite the cols array, removing duplicates
     do rp = rstart, rend-1
        if (nzeros(cols(rp)) == 0) then
           cols(index) = cols(rp)
           nzeros(cols(index)) = 1
           index = index + 1
        end if
     end do

     ! Set the new end location for the row
     rowp(i+1) = index

     ! Reset the array of flags
     do rp = rowp(i), index-1
        nzeros(cols(rp)) = 0
     end do
     rstart = rend

     call quicksortArray(cols(rowp(i):rowp(i+1)-1))
  end do

end subroutine computeKmatNzPattern

subroutine computeDmatNzPattern(ne, ntw, tconn, ncols, rowp, cols, info)
  ! Compute the non-zero pattern of the second-derivative matrix.  The
  ! D matrix is the second derivative of the barrier term w.r.t. x.
  ! The non-zero pattern is the non-zero pattern contributed by the
  ! filter.
  !
  ! Input:
  ! ne:      the number of elements
  ! ntw:     the number of element weights
  ! ncols:   the number of entries in the D matrix
  !
  ! Output:
  ! rowp:    pointer into the cols array
  ! cols:    column indices
  ! info:    did this work or not?

  use precision
  use quicksort
  implicit none

  ! Input data
  integer, intent(in) :: ne, ntw, tconn(ntw,ne), ncols
  integer, intent(inout) :: rowp(ne+1), cols(ncols)
  integer, intent(out) :: info

  ! Temporary data for the function
  integer :: i, j, k, nentry
  integer :: jp, jstart, jend, nzeros(ne)

  ! Zero all the entries in the row pointer
  rowp(:) = 0

  ! Compute the number of entries in each row
  do i = 1, ne
     nentry = 0
     do j = 1, ntw
        if (tconn(j,i) > 0) then
           nentry = nentry + 1
        end if
     end do

     do j = 1, ntw
        if (tconn(j,i) > 0) then
           rowp(1+tconn(j,i)) = rowp(1+tconn(j,i)) + nentry
        end if
     end do
  end do

  ! Convert this into a pointer array
  rowp(1) = 1
  do i = 1, ne
     rowp(i+1) = rowp(i) + rowp(i+1)
  end do

  ! Check if the array is large enough
  if (ncols < rowp(ne+1)-1) then
     info = rowp(ne+1)-1
     return
  end if

  ! Success, now sort the array
  info = 0

  ! Add the non-zero pattern from each element
  do i = 1, ne
     do j = 1, ntw
        if (tconn(j,i) > 0) then
           do k = 1, ntw
              if (tconn(k,i) > 0) then
                 jp = tconn(k,i)
                 cols(rowp(jp)) = tconn(j,i)
                 rowp(jp) = rowp(jp)+1
              end if
           end do
        end if
     end do
  end do

  ! Reset the pointer array
  do i = ne, 1, -1
     rowp(i+1) = rowp(i)
  end do
  rowp(1) = 1

  ! Now, we've over-counted the entries, remove duplicates in each
  ! row. Note that this is an O(n) run time, but also has an O(n)
  ! storage requirement.
  nzeros(:) = 0

  k = 1
  jstart = 1
  do i = 1, ne
     jend = rowp(i+1)

     ! Overwrite the cols array, removing duplicates
     do jp = jstart, jend-1
        if (nzeros(cols(jp)) == 0) then
           cols(k) = cols(jp)
           nzeros(cols(k)) = 1
           k = k + 1
        end if
     end do

     ! Set the new end location for the row
     rowp(i+1) = k

     ! Reset the array of flags
     do jp = rowp(i), k-1
        nzeros(cols(jp)) = 0
     end do
     jstart = jend

     call quicksortArray(cols(rowp(i):rowp(i+1)-1))
  end do

end subroutine computeDmatNzPattern

subroutine vecDot(nb, n, x, y, vdot)
  ! Compute the dot product of the two vectors x/y
  !
  ! Input:
  ! nb:  the block size
  ! n:   the number of nodes in the vectors
  ! x:   the first vector
  ! y:   the second vector
  !
  ! Output:
  ! vdot: dot(x, y)

  use precision
  implicit none

  integer, intent(in) :: nb, n
  real(kind=dtype), intent(in) :: x(nb,n), y(nb,n)
  real(kind=dtype), intent(out) :: vdot

  integer :: i

  vdot = 0.0_dtype
  do i = 1, n
     vdot = vdot + dot_product(x(:,i), y(:,i))
  end do

end subroutine vecDot

subroutine matMult(nb, n, m, ncols, rowp, cols, A, x, y)
  ! Multiply a block-compressed sparse row matrix by a corresponding
  ! vector
  !
  ! Input:
  ! nb:     the block size
  ! n:      the row dimension of the matrix
  ! ncols:  the length of the cols array
  ! rowp:   the length of the row pointer array
  ! A:      the A matrix
  ! x:      x-input vector
  !
  ! Output:
  ! y:      the y-vector

  use precision
  implicit none

  ! Input/output definitions
  integer, intent(in) :: nb, n, m, ncols, rowp(n+1), cols(ncols)
  real(kind=dtype), intent(in) :: x(nb,m), A(nb,nb,ncols)
  real(kind=dtype), intent(inout) :: y(nb,n)

  ! Local internal data
  integer :: i, j, jp

  do i = 1, n
     y(:,i) = 0.0_dtype
     do jp = rowp(i), rowp(i+1)-1
        j = cols(jp)
        y(:,i) = y(:,i) + matmul(A(:,:,jp), x(:,j))
     end do
  end do

end subroutine matMult

subroutine computeNzMatMultInner( &
     np, mp, npcols, prowp, pcols, &
     na, nacols, arowp, acols, &
     pcolp, prows, nccols, crowp, ccols, info)
  ! Compute the non-zero pattern of the C = P*A*P^{T} matrix.  The
  ! first step is to compute the non-zero pattern of the
  ! compressed-sparse column representation of P. Next, this CSC
  ! data structure is used to facilitate the construction of the
  ! non-zero pattern of the C matrix.
  !
  ! Input:
  ! np, mp:   the number of rows/columns in the P matrix
  ! npcols:   the length of the pcols/prows array
  ! prowp:    pointer into the pcols array for each row
  ! pcols:    the column indices of the matrix P
  ! na:       the number of rows in the matrix A
  ! nacols:   the length of the array acols
  ! arowp:    pointer into the acols array for each row
  ! acols:    the column indices in the matrix A
  ! nccols:   the length of the ccols array - this is an estimate
  !
  ! Output:
  ! pcolp:    pointer into the prows array for each column
  ! prows:    the row indices for each column in P
  ! crowp:    the output C row pointer
  ! ccols:    the output C column indices
  ! info:     flag to indicate whether we had a failure or not

  use quicksort
  implicit none

  ! The input/ouput data
  integer, intent(in) :: np, mp, npcols, prowp(np+1), pcols(npcols)
  integer, intent(in) :: na, nacols, arowp(na+1), acols(nacols)
  integer, intent(inout) :: pcolp(mp+1), prows(npcols)
  integer, intent(in) :: nccols
  integer, intent(inout) :: crowp(np+1), ccols(nccols)
  integer, intent(out) :: info

  ! The internal data
  integer :: i, j, k, l, jp, kp, lp
  integer :: ncvals, nzeros(mp)

  ! Unless this flag is set, then we have success
  info = 0

  ! Obtain the equivalent CSC format for P
  call computeCSRToCSC(np, mp, npcols, prowp, pcols, pcolp, prows)

  ! Now, start to compute the non-zero pattern
  ncvals = 1
  crowp(1) = 1

  ! Zero all the entries in the non-zero array
  nzeros(:) = 0

  do i = 1, np
     ! Loop over the row P[i,:]
     do jp = prowp(i), prowp(i+1)-1
        j = pcols(jp)

        ! Loop over the row A[j,:]
        do kp = arowp(j), arowp(j+1)-1
           k = acols(kp)

           ! Loop over the column P[:,k]
           do lp = pcolp(k), pcolp(k+1)-1
              l = prows(lp)

              ! Flag entries that are non-zero
              if (nzeros(l) == 0) then
                 nzeros(l) = 1
                 ccols(ncvals) = l
                 ncvals = ncvals + 1

                 ! The array is not long enough, give up!
                 if (ncvals > nccols) then
                    info = -1
                    return
                 end if
              end if
           end do
        end do
     end do

     ! Reset the array of flags
     do jp = crowp(i), ncvals-1
        nzeros(ccols(jp)) = 0
     end do

     ! Set the pointer to the end of the array
     crowp(i+1) = ncvals

     ! Sort the rows of the matrix
     call quicksortArray(ccols(crowp(i):crowp(i+1)-1))
  end do

end subroutine computeNzMatMultInner

subroutine computeMatMultInner(nb, &
     np, npcols, prowp, pcols, P, &
     na, nacols, arowp, acols, A, &
     nc, nccols, crowp, ccols, C)
  ! Compute the inner product of the interpolant matrix P with the the
  ! matrix A:
  !
  ! C = P*A*P^{T}
  !
  ! Note that P is stored as a compressed-sparse matrix in component
  ! form while A and C are stored in a block-compressed-row format.
  ! In index form, this inner product computation takes the following
  ! form:
  !
  ! C_{ij} = P_{ik} A_{kl} P_{jl}
  !        = P_{ik} sdot(A_{k:}, P_{j:})
  !
  ! Where sdot() is a sparse vector-vector product. This product is
  ! performed by the code: addSparseMatRowDot.
  !
  ! Input:
  ! nb:          the block size
  ! np, npcols:  the number of rows and length of the pcols array
  ! prowp:       the row-pointer array for P
  ! pcols:       the column indices of P
  ! P:           the values in the P matrix
  ! na, nacols:  the number of rows and length of the acols array
  ! arowp:       the row-pointer array for A
  ! acols:       the column indices of A
  ! A:           the values in the A matrix
  ! nc, nccols:  the number of rows and length of the ccols array
  ! crowp:       the row-pointer array for C
  ! ccols:       the column indices of C
  !
  ! Output:
  ! C:           the values in the C matrix

  use precision
  implicit none

  integer, intent(in) :: nb
  integer, intent(in) :: np, npcols, prowp(np+1), pcols(npcols)
  integer, intent(in) :: na, nacols, arowp(na+1), acols(nacols)
  integer, intent(in) :: nc, nccols, crowp(nc+1), ccols(nccols)
  real(kind=dtype), intent(in) :: P(npcols), A(nb,nb,nacols)
  real(kind=dtype), intent(inout) :: C(nb,nb,nccols)

  ! Local data storage
  integer :: i, j, k, jp, kp

  ! Zero the entries in the matrix
  C(:,:,:) = 0.0_dtype

  ! Loop over the non-zero components of the C matrix
  do i = 1, nc
     do jp = crowp(i), crowp(i+1)-1
        j = ccols(jp)

        ! Loop over the non-zero entries in the i-th row of P
        do kp = prowp(i), prowp(i+1)-1
           k = pcols(kp)

           ! Compute a sparse dot-product of row A[k,:] with P[j,:]
           ! This is why the entries must be sorted - for efficiency!
           call addSparseMatRowDot(P(kp), nb, &
                nacols, arowp(k), arowp(k+1), acols, A, &
                npcols, prowp(j), prowp(j+1), pcols, P, C(:,:,jp))
        end do
     end do
  end do

end subroutine computeMatMultInner

subroutine addSparseMatRowDot(alpha, nb, &
     nacols, istart, iend, acols, A, &
     nbcols, jstart, jend, bcols, B, D)
  ! Add the inner product of two sparse rows of a matrix
  ! together. This is used for matrix-vector products. The column
  ! indices in each row must be sorted in ascending order for this to
  ! work properly.
  !
  ! Input:
  ! nacols:  the length of the acols array
  ! istart:  the start index of the A row
  ! iend:    the beginning of the next A row
  ! acols:   the column indices in A
  ! A:       the values of the A matrix
  ! nbcols:  the length of the bcols array
  ! jstart:  the start index of the B row
  ! jend:    the beginning of the next B row
  ! bcols:   the columns indices in the B matrix
  ! B:       the values in B
  !
  ! Output:
  ! D:       the additional inner product

  use precision
  implicit none

  integer, intent(in) :: nb
  integer, intent(in) :: nacols, nbcols, istart, jstart, iend, jend
  integer, intent(in) :: acols(nacols), bcols(nbcols)
  real(kind=dtype), intent(in) :: alpha, A(nb,nb,nacols), B(nbcols)
  real(kind=dtype), intent(inout) :: D(nb,nb)

  ! The local integer information
  integer :: ip, jp

  ! Start from the beginning
  jp = jstart
  do ip = istart, iend-1
     do while (jp <= jend-1 .and. bcols(jp) < acols(ip))
        jp = jp + 1
     end do

     ! Exit if the pointer exceeded the last row
     if (jp >= jend) then
        exit
     end if

     if (acols(ip) == bcols(jp)) then
        D(:,:) = D(:,:) + alpha*B(jp)*A(:,:,ip)
     end if
  end do

end subroutine addSparseMatRowDot

subroutine computeRestrict(nb, np, mp, npcols, prowp, pcols, P, uf, uc)
  ! Compute the restriction of the solution on the finer grid, uf, to
  ! the solution on the coarser grid, uc.
  !
  ! Input:
  ! nb:          the block size
  ! np, npcols:  the number of rows and length of the pcols array
  ! prowp:       the row-pointer array for P
  ! pcols:       the column indices of P
  ! P:           the values in the P matrix
  ! uf:          the solution on the finer grid
  !
  ! Output:
  ! uc:          the restricted solution on the coarser grid

  use precision
  implicit none

  integer, intent(in) :: nb
  integer, intent(in) :: np, mp, npcols, prowp(np+1), pcols(npcols)
  real(kind=dtype), intent(in) :: P(npcols), uf(nb,mp)
  real(kind=dtype), intent(inout) :: uc(nb,np)

  integer :: i, j, jp

  do i = 1, np
     do jp = prowp(i), prowp(i+1)-1
        j = pcols(jp)
        uc(:,i) = uc(:,i) + P(jp)*uf(:,j)
     end do
  end do

end subroutine computeRestrict

subroutine computeInterp(nb, np, mp, npcols, prowp, pcols, P, uc, uf)
  ! Compute the interpolation of the solution from the coarse mesh
  ! to the finer mesh.
  !
  ! Input:
  ! nb:          the block size
  ! np, npcols:  the number of rows and length of the pcols array
  ! prowp:       the row-pointer array for P
  ! pcols:       the column indices of P
  ! P:           the values in the P matrix
  ! uc:          the restricted solution on the coarser grid
  !
  ! Output:
  ! uf:          the solution on the finer grid

  use precision
  implicit none

  integer, intent(in) :: nb
  integer, intent(in) :: np, mp, npcols, prowp(np+1), pcols(npcols)
  real(kind=dtype), intent(in) :: P(npcols), uc(nb,np)
  real(kind=dtype), intent(inout) :: uf(nb,mp)

  integer :: i, jp, j

  do i = 1, np
     do jp = prowp(i), prowp(i+1)-1
        j = pcols(jp)
        uf(:,j) = uf(:,j) + P(jp)*uc(:,i)
     end do
  end do

end subroutine computeInterp

subroutine applyVecBCs(nb, n, nbc, bcnodes, bcvars, U)
  ! Apply boundary conditions to the given vector
  !
  ! Input:
  ! nb:       the block size
  ! n:        the number of unknowns
  ! nbc:      the number of boundary conditions
  ! bcnodes:  the node numbers associated with the boundary conditions
  ! bcvars:   the varaibles to be zeroed
  !
  ! Input/output:
  ! U:        the displacement (or equivalent) vector

  use precision
  implicit none

  ! Input/output specification
  integer, intent(in) :: nb, n, nbc, bcnodes(nbc), bcvars(nb,nbc)
  real(kind=dtype), intent(inout) :: U(nb,n)

  ! Local data
  integer :: ibc, i, k

  do ibc = 1, nbc
     i = bcnodes(ibc)
     do k = 1, nb
        if (bcvars(k,ibc) > 0) then
           U(k,i) = 0.0_dtype
        end if
     end do
  end do

end subroutine applyVecBCs

subroutine applyMatBCs(nb, n, nbc, bcnodes, bcvars, &
     ncols, rowp, cols, A, ident)
  ! Apply the boundary conditions to the stiffness matrix by
  ! zeroing-out the rows and setting the diagonals of those rows to
  ! the identity.
  !
  ! Input:
  ! nb:       the block size
  ! n:        the number of unknowns
  ! nbc:      the number of boundary conditions
  ! bcnodes:  the node numbers associated with the boundary conditions
  ! bcvars:   the varaibles to be zeroed
  ! ncols:   the length of the cols array
  ! rowp:    the row pointer array
  ! cols:    the column indices in each row
  ! ident:   flag to indicate whether to use the identity on the diagonal
  !
  ! Output:
  ! A:       the modified stiffness matrix

  use precision
  implicit none

  ! Input/output data
  integer, intent(in) :: nb, n, nbc, bcnodes(nbc), bcvars(nb,nbc)
  integer, intent(in) :: ncols, rowp(n+1), cols(ncols)
  logical, intent(in) :: ident
  real(kind=dtype), intent(inout) :: A(nb,nb,ncols)

  ! Local data
  integer :: ibc, i, j, jp, k, kp

  ! Apply the boundary conditions
  do ibc = 1, nbc
     i = bcnodes(ibc)
     do jp = rowp(i), rowp(i+1)-1
        j = cols(jp)

        do k = 1, nb
           if (bcvars(k,ibc) > 0) then
              ! Zero this row
              A(k,:,jp) = 0.0_dtype
              A(:,k,jp) = 0.0_dtype
           end if
        end do

        if (i == j .and. ident) then
           ! Set the identity matrix on the diagonal
           do k = 1, nb
              if (bcvars(k,ibc) > 0) then
                 A(k,k,jp) = 1.0_dtype
              end if
           end do
        else
           ! Zero the corresponding column of the matrix
           do kp = rowp(j), rowp(j+1)-1
              if (cols(kp) == i) then
                 ! Zero the corresponding components
                 do k = 1, nb
                    if (bcvars(k,ibc) > 0) then
                       ! Zero this row
                       A(k,:,kp) = 0.0_dtype
                       A(:,k,kp) = 0.0_dtype
                    end if
                 end do
              end if
           end do
        end if
     end do
  end do

end subroutine applyMatBCs

subroutine computeDiagFactor(nb, n, ncols, rowp, cols, A, D)
  ! Extract the diagonal entry of the matrix and compute its inverse
  ! for later usage.
  !
  ! Input:
  ! nb:      the block size
  ! n:       the number of unknowns
  ! ncols:   the length of the cols array
  ! rowp:    the pointer to the beginning of each row
  ! cols:    the column index of each entry
  ! K:       the stiffness matrix
  !
  ! Output:
  ! D:       the factored diagonal entries

  use precision
  implicit none

  ! The input declarations
  integer, intent(in) :: nb, n, ncols, rowp(n+1), cols(ncols)
  real(kind=dtype), intent(in) :: A(nb,nb,ncols)
  real(kind=dtype), intent(inout) :: D(nb,nb,n)

  ! Internal data
  integer :: i, jp, j, k, piv(nb), info
  real(kind=dtype) :: work(nb*nb)

  ! Zero all entries in the matrix
  D(:,:,:) = 0.0_dtype

  do i = 1, n
     ! Scan through the row until we find the diagonal entry, then
     ! compute the inverse and exit
     do jp = rowp(i), rowp(i+1)-1
        if (cols(jp) == i) then
           ! Copy the matrix to D then factor it and compute the inverse
           D(:,:,i) = A(:,:,jp)
           call dsytrf('U', nb, D(:,:,i), nb, piv, work, nb*nb, info)
           call dsytri('U', nb, D(:,:,i), nb, piv, work, info)

           ! Copy over the remainder of D^{-1} to the lower triangular part
           do j = 1, nb
              do k = 1, j-1
                 D(j,k,i) = D(k,j,i)
              end do
           end do

           ! Quit to start the next row
           exit
        end if
     end do
  end do

end subroutine computeDiagFactor

subroutine applySOR(nb, n, ncols, rowp, cols, A, D, niters, omega, b, u)
  ! Apply a specified number of iterations of SOR.
  !
  ! Input:
  ! n:       the number of unknowns
  ! ncols:   the length of the cols array
  ! rowp:    the pointer to the beginning of each row
  ! cols:    the column index of each entry
  ! K:       the stiffness matrix
  ! D:       the factored diagonal entries
  ! niters:  the number of iterations to apply
  ! omega:   the relaxation factor
  ! b:       the right-hand-side
  !
  ! Output:
  ! u:       the solution vectors

  use precision
  implicit none

  ! The input declarations
  integer, intent(in) :: nb, n, ncols, rowp(n+1), cols(ncols), niters
  real(kind=dtype), intent(in) :: A(nb,nb,ncols), D(nb,nb,n), b(nb,n)
  real(kind=dtype), intent(in) :: omega
  real(kind=dtype), intent(inout) :: u(nb,n)

  ! Temporary storage for each row
  real(kind=dtype) :: t(nb)
  integer :: i, j, jp, jstart, jend, k
  integer :: neven

  ! If the remainder is 1 then n is odd
  neven = n - mod(n,2)

  do k = 1, niters
     ! Apply a foward pass of SOR
     do i = 1, n, 2
        t(:) = b(:,i)

        jstart = rowp(i)
        jend = rowp(i+1)-1
        do jp = jstart, jend
           j = cols(jp)

           if (i .ne. j) then
              t = t - matmul(A(:,:,jp), u(:,j))
           end if
        end do

        u(:,i) = (1.0 - omega)*u(:,i) + omega*matmul(D(:,:,i), t)
     end do

     do i = neven, 2, -2
        t(:) = b(:,i)

        jstart = rowp(i)
        jend = rowp(i+1)-1
        do jp = jstart, jend
           j = cols(jp)

           if (i .ne. j) then
              t = t - matmul(A(:,:,jp), u(:,j))
           end if
        end do

        u(:,i) = (1.0 - omega)*u(:,i) + omega*matmul(D(:,:,i), t)
     end do
  end do

end subroutine applySOR

subroutine addDiagonal(nb, ne, diag, ncols, rowp, cols, D)
  ! Add the given terms to the diagonal
  !
  ! Input:
  ! ne:       the number of elements
  ! diag:     the diagonal entries to add
  ! ncols:    the number of entries in the matrix
  ! rowp:     the row pointer
  ! cols:     the column indices
  !
  ! Output:
  ! D:        the matrix values

  use precision
  implicit none

  ! Input/output
  integer, intent(in) :: nb, ne
  real(kind=dtype), intent(in) :: diag(nb,ne)
  integer, intent(in) :: ncols, rowp(ne+1), cols(ncols)
  real(kind=dtype), intent(inout) :: D(nb,nb,ncols)

  ! Temp data
  integer :: i, k, jp

  do i = 1, ne
     do jp = rowp(i), rowp(i+1)-1
        if (cols(jp) == i) then
           ! Add the digaonal entries
           do k = 1, nb
              D(k,k,jp) = D(k,k,jp) + diag(k,i)
           end do

           exit
        end if
     end do
  end do

end subroutine addDiagonal

subroutine computeInnerNull(nb, n, Z, ncols, rowp, cols, A, D)
  ! Reduce the matrix
  !
  ! Input:
  ! nb:     the block size for the A matrix
  ! n:      the number of variables
  ! Z:      the null space basis
  ! ncols:  the number of entries in the A/D matrices
  ! rowp:   the pointer into the rows of the matrix
  ! cols:   the column indices
  ! A:      the original matrix
  !
  ! Output:
  ! D:      the output matrix

  use precision
  implicit none

  integer, intent(in) :: nb, n
  real(kind=dtype), intent(in) :: Z(nb,nb-1,n)
  integer, intent(in) :: ncols, rowp(n+1), cols(ncols)
  real(kind=dtype), intent(in) :: A(nb,nb,ncols)
  real(kind=dtype), intent(inout) :: D(nb-1,nb-1,ncols)

  integer :: i, j, jp

  ! Evaluate the inner product with each block in the matrix
  do i = 1, n
     do jp = rowp(i), rowp(i+1)-1
        j = cols(jp)
        D(:,:,jp) = matmul(transpose(Z(:,:,i)), matmul(A(:,:,jp), Z(:,:,j)))
     end do
  end do

end subroutine computeInnerNull
