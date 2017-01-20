! The following code implements a full space method for
! stress-constrained topology optimization. These subroutines should
! be called from python once they are wrapped with f2py.
!
! Copyright (c) Graeme J. Kennedy 2015, All rights reserved.

subroutine computeMass(n, ne, ntw, nmat, &
     conn, X, tconn, tweights, xdv, rho, mass)
  ! Compute the mass of the structure given the material densities
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! rho:      the density values within each element
  !
  ! Output:
  ! mass:     the mass

  use precision
  implicit none

  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: rho(nmat)
  real(kind=dtype), intent(out) :: mass

  ! Temporary data used internally
  integer :: i, j, k, kp
  real(kind=dtype) :: area, ttmp, xtmp

  ! Zero the initial mass
  mass = 0.0_dtype

  ! Loop over all elements within the mesh
  do i = 1, ne
     ! Compute the area
     area = 0.5*( &
          (X(1,conn(2,i)) + X(1,conn(1,i)))*(X(2,conn(2,i)) - X(2,conn(1,i))) + &
          (X(1,conn(4,i)) + X(1,conn(2,i)))*(X(2,conn(4,i)) - X(2,conn(2,i))) + &
          (X(1,conn(3,i)) + X(1,conn(4,i)))*(X(2,conn(3,i)) - X(2,conn(4,i))) + &
          (X(1,conn(1,i)) + X(1,conn(3,i)))*(X(2,conn(1,i)) - X(2,conn(3,i))))

     ! Compute the filtered thickness value
     ttmp = 0.0_dtype
     do k = 1, ntw
        kp = tconn(k,i)
        if (kp > 0) then
           ttmp = ttmp + tweights(k,i)/xdv(1,kp)
        end if
     end do

     do j = 1, nmat
        ! Compute the filtered material variable
        xtmp = 0.0_dtype
        do k = 1, ntw
           kp = tconn(k,i)
           if (kp > 0) then
              xtmp = xtmp + tweights(k,i)/xdv(1+j,kp)
           end if
        end do

        mass = mass + area*rho(j)*xtmp*ttmp
     end do
  end do

end subroutine computeMass

subroutine addMassDeriv(n, ne, ntw, nmat, &
     conn, X, tconn, tweights, xdv, rho, mass)
  ! Compute the mass of the structure given the material densities
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! rho:      the density values within each element
  !
  ! Output:
  ! mass:     the derivative of the mass

  use precision
  implicit none

  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: rho(nmat)
  real(kind=dtype), intent(inout) :: mass(nmat+1,ne)

  ! Temporary data used internally
  integer :: i, j, k, kp
  real(kind=dtype) :: area, ttmp, xtmp

  ! Loop over all elements within the mesh
  do i = 1, ne
     ! Compute the area
     area = 0.5*( &
          (X(1,conn(2,i)) + X(1,conn(1,i)))*(X(2,conn(2,i)) - X(2,conn(1,i))) + &
          (X(1,conn(4,i)) + X(1,conn(2,i)))*(X(2,conn(4,i)) - X(2,conn(2,i))) + &
          (X(1,conn(3,i)) + X(1,conn(4,i)))*(X(2,conn(3,i)) - X(2,conn(4,i))) + &
          (X(1,conn(1,i)) + X(1,conn(3,i)))*(X(2,conn(1,i)) - X(2,conn(3,i))))

     ! Compute the filtered thickness value
     ttmp = 0.0_dtype
     do k = 1, ntw
        kp = tconn(k,i)
        if (kp > 0) then
           ttmp = ttmp + tweights(k,i)/xdv(1,kp)
        end if
     end do

     do j = 1, nmat
        ! Compute the filtered material variable
        xtmp = 0.0_dtype
        do k = 1, ntw
           kp = tconn(k,i)
           if (kp > 0) then
              xtmp = xtmp + tweights(k,i)/xdv(1+j,kp)
           end if
        end do

        ! Add the derivative of the mass
        do k = 1, ntw
           kp = tconn(k,i)
           if (kp > 0) then
              mass(1,kp) = mass(1,kp) - area*rho(j)*xtmp*tweights(k,i)/xdv(1,kp)**2
              mass(1+j,kp) = mass(1+j,kp) - area*rho(j)*ttmp*tweights(k,i)/xdv(1+j,kp)**2
           end if
        end do
     end do
  end do

end subroutine addMassDeriv

subroutine addMass2ndDeriv(n, ne, ntw, nmat, &
     conn, X, tconn, tweights, xdv, rho, ncols, rowp, cols, D)
  ! Compute the mass of the structure given the material densities
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! rho:      the density values within each element
  ! ncols:    the length of the columns array
  ! rowp:     the row pointer
  ! cols:     the column index
  !
  ! Output:
  ! D:      the matrix of second derivatives w.r.t. x

  use precision
  implicit none

  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: rho(nmat)
  integer, intent(in) :: ncols, rowp(ne+1), cols(ncols)
  real(kind=dtype), intent(inout) :: D(1+nmat,1+nmat,ncols)

  ! Temporary data used internally
  integer :: i, j, k, ki, kj, ii, jj, jp, kp
  real(kind=dtype) :: area, tw, tmp(1+nmat)

  ! Loop over all elements within the mesh
  do i = 1, ne
     ! Compute the area
     area = 0.5*( &
          (X(1,conn(2,i)) + X(1,conn(1,i)))*(X(2,conn(2,i)) - X(2,conn(1,i))) + &
          (X(1,conn(4,i)) + X(1,conn(2,i)))*(X(2,conn(4,i)) - X(2,conn(2,i))) + &
          (X(1,conn(3,i)) + X(1,conn(4,i)))*(X(2,conn(3,i)) - X(2,conn(4,i))) + &
          (X(1,conn(1,i)) + X(1,conn(3,i)))*(X(2,conn(1,i)) - X(2,conn(3,i))))

     ! Compute the filtered thickness/material variable values
     tmp(:) = 0.0_dtype
     do k = 1, ntw
        kp = tconn(k,i)
        if (kp > 0) then
           tmp(:) = tmp(:) + tweights(k,i)/xdv(:,kp)
        end if
     end do

     do ii = 1, ntw
        ki = tconn(ii,i)
        if (ki > 0) then
           do jj = 1, ntw
              kj = tconn(jj,i)
              if (kj > 0) then
                 do jp = rowp(ki), rowp(ki+1)-1
                    if (cols(jp) == kj) then
                       tw = area*tweights(ii,i)*tweights(jj,i)

                       ! Add the off-diagonal terms
                       do j = 1, nmat
                          D(1,1+j,jp) = D(1,1+j,jp) + &
                               rho(j)*tw/(xdv(1,ki)*xdv(1+j,kj))**2
                          D(1+j,1,jp) = D(1+j,1,jp) + &
                               rho(j)*tw/(xdv(1+j,ki)*xdv(1,kj))**2
                       end do

                       ! Add the terms to the diagonal
                       if (ii == jj) then
                          do j = 1, nmat
                             D(1,1,jp) = D(1,1,jp) + &
                                  2.0*rho(j)*area*tweights(ii,i)*tmp(1+j)/xdv(1,ki)**3
                             D(1+j,1+j,jp) = D(1+j,1+j,jp) + &
                                  2.0*rho(j)*area*tweights(ii,i)*tmp(1)/xdv(1+j,ki)**3
                          end do
                       end if

                       exit
                    end if
                 end do
              end if
           end do
        end if
     end do
  end do
end subroutine addMass2ndDeriv

subroutine computeConstitutiveMat(index, ptype, ne, ntw, nmat, &
     tconn, tweights, xdv, qval, qxval, Cmat, Celem)
  ! Compute the constitutive matrix for the given element index using
  ! the specified parameterization - either a continuous thickness or
  ! a RAMP interpolation.
  !
  ! Input:
  ! index:    the element index
  ! ptype:    the type of stiffness parametrization
  ! ne:       the number of elements
  ! ntw:      the filter size
  ! nmat:     the number of elements
  ! tconn:    the connectivity of the filter
  ! tweights: the filter weights
  ! xdv:      the design variables
  ! qval:     the RAMP penalty value
  ! qxval:    the RAMP penalty for the material parametrization
  ! Cmat:     the material selection variables
  !
  ! Output:
  ! Celem:    the element constitutive object

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: index, ptype, ne, ntw, nmat, tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: qval, qxval, Cmat(3,3,nmat)
  real(kind=dtype), intent(inout) :: Celem(3,3)

  ! Temporary data used in this function
  integer :: i, j
  real(kind=dtype) :: ttmp, xtmp, tpenalty, penalty

  ! Set the parameters for this function
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero the stiffness matrix
  Celem(:,:) = 0.0_dtype

  ! Compute the filtered thickness
  ttmp = 0.0_dtype
  do j = 1, ntw
     if (tconn(j,index) > 0) then
        ttmp = ttmp + tweights(j,index)/xdv(1,tconn(j,index))
     end if
  end do

  ! Compute the thickness penalty factor
  tpenalty = ttmp
  if (ptype == 2) then
     call computePenalty(ttmp, qval, tpenalty)
  end if

  do i = 1, nmat
     ! Compute the filtered material variable
     xtmp = 0.0_dtype
     do j = 1, ntw
        if (tconn(j,index) > 0) then
           xtmp = xtmp + tweights(j,index)/xdv(1+i,tconn(j,index))
        end if
     end do

     call computePenalty(xtmp, qxval, penalty)
     Celem(:,:) = Celem(:,:) + tpenalty*penalty*Cmat(:,:,i)
  end do

end subroutine computeConstitutiveMat

subroutine addConstitutiveMatDeriv(index, ptype, ne, ntw, nmat, &
     tconn, tweights, xdv, qval, qxval, inner, rx)
  ! Add the inner product of the constitutive matrix with a given
  ! vectors to the given vector. Note that the derivative of this
  ! vector will depend on the
  !
  ! Input:
  ! index:    the element index
  ! ptype:    the type of stiffness parametrization
  ! ne:       the number of elements
  ! ntw:      the filter size
  ! nmat:     the number of elements
  ! tconn:    the connectivity of the filter
  ! tweights: the filter weights
  ! xdv:      the design variables
  ! qval:     the RAMP penalty value
  ! qxval:    the RAMP penalty for the material parametrization
  ! inner:    the inner product with the full constitutive matrix
  !
  ! Output:
  ! rx:       the derivative

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: index, ptype, ne, ntw, nmat, tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: qval, qxval
  real(kind=dtype), intent(in) :: inner(nmat)
  real(kind=dtype), intent(inout) :: rx(nmat+1,ne)

  ! Temporary data used in this function
  integer :: i, j, k
  real(kind=dtype) :: ttmp, xtmp, tpenalty, penalty
  real(kind=dtype) :: deriv, tderiv

  ! Set the parameters for this function
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Compute the filtered thickness
  ttmp = 0.0_dtype
  do j = 1, ntw
     if (tconn(j,index) > 0) then
        ttmp = ttmp + tweights(j,index)/xdv(1,tconn(j,index))
     end if
  end do

  ! Compute the thickness penalty factor
  tpenalty = ttmp
  tderiv = 1.0_dtype
  if (ptype == 2) then
     call computePenaltyDeriv(ttmp, qval, tpenalty, tderiv)
  end if

  do i = 1, nmat
     ! Compute the filtered material variable
     xtmp = 0.0_dtype
     do j = 1, ntw
        if (tconn(j,index) > 0) then
           xtmp = xtmp + tweights(j,index)/xdv(1+i,tconn(j,index))
        end if
     end do

     ! Compute the penalty for the material selection variables
     call computePenaltyDeriv(xtmp, qxval, penalty, deriv)

     ! Add up the contributions from each donnor cell
     do j = 1, ntw
        k = tconn(j,index)
        if (k > 0) then
           ! Compute the contribution to the thickness parameter
           rx(1,k) = rx(1,k) &
                - inner(i)*penalty*tderiv*tweights(j,index)/xdv(1,k)**2

           ! Compute the contribution to the material parameter
           rx(1+i,k) = rx(1+i,k) &
                - inner(i)*tpenalty*deriv*tweights(j,index)/xdv(1+i,k)**2
        end if
     end do
  end do

end subroutine addConstitutiveMatDeriv

subroutine addConstitutiveMatDerivPerb( &
     index, ptype, ne, ntw, nmat, &
     tconn, tweights, xdv, xperb, qval, qxval, Cmat, Celem)
  ! Add the inner product of the constitutive matrix with a given
  ! vectors to the given vector. Note that the derivative of this
  ! vector will depend on the
  !
  ! Input:
  ! index:    the element index
  ! ptype:    the type of stiffness parametrization
  ! ne:       the number of elements
  ! ntw:      the filter size
  ! nmat:     the number of elements
  ! tconn:    the connectivity of the filter
  ! tweights: the filter weights
  ! xdv:      the design variables
  ! xperb:    the perturbation of the design variables
  ! qval:     the RAMP penalty value
  ! qxval:    the RAMP penalty for the material parametrization
  ! Cmat:     the material selection variables
  !
  ! Output:
  ! Celem:    the element constitutive object

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: index, ptype, ne, ntw, nmat, tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: xperb(nmat+1,ne), qval, qxval
  real(kind=dtype), intent(in) :: Cmat(3,3,nmat)
  real(kind=dtype), intent(inout) :: Celem(3,3)

  ! Temporary data used in this function
  integer :: i, j, kj
  real(kind=dtype) :: ttmp, dttmp, xtmp, dxtmp
  real(kind=dtype) :: tpenalty, penalty
  real(kind=dtype) :: deriv, tderiv

  ! Set the parameters for this function
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero the derivattive of the constitutive matrix
  Celem(:,:) = 0.0_dtype

  ! Compute the filtered thickness
  ttmp = 0.0_dtype
  dttmp = 0.0_dtype
  do j = 1, ntw
     kj = tconn(j,index)
     if (kj > 0) then
        ttmp = ttmp + tweights(j,index)/xdv(1,kj)
        dttmp = dttmp - xperb(1,kj)*tweights(j,index)/xdv(1,kj)**2
     end if
  end do

  ! Compute the thickness penalty factor
  tpenalty = ttmp
  tderiv = dttmp
  if (ptype == 2) then
     call computePenaltyDeriv(ttmp, qval, tpenalty, tderiv)
     tderiv = tderiv*dttmp
  end if

  do i = 1, nmat
     ! Compute the filtered material variable
     xtmp = 0.0_dtype
     dxtmp = 0.0_dtype
     do j = 1, ntw
        kj = tconn(j,index)
        if (kj > 0) then
           xtmp = xtmp + tweights(j,index)/xdv(1+i,kj)
           dxtmp = dxtmp - xperb(1+i,kj)*tweights(j,index)/xdv(1+i,kj)**2
        end if
     end do

     ! Compute the penalty for the material selection variables
     call computePenaltyDeriv(xtmp, qxval, penalty, deriv)
     deriv = deriv*dxtmp

     ! Add up the contribution from this material
     Celem(:,:) = Celem(:,:) + (tderiv*penalty + tpenalty*deriv)*Cmat(:,:,i)
  end do

end subroutine addConstitutiveMatDerivPerb

subroutine computeKmat( &
     ptype, n, ne, ntw, nmat, conn, X, &
     tconn, tweights, xdv, qval, qxval, Cmat, &
     ncols, rowp, cols, K)
  ! Compute the global stiffness matrix and store it in the given
  ! compressed sparse row data format.
  !
  ! Input:
  ! ptype:    the type of stiffness parametrization
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the max size of the filter
  ! nmat:     the number of material selections
  ! conn:     the element connectivity
  ! X:        the nodal locations
  ! tconn:    the thickness to design variable weight array
  ! tweights: the thickness weights
  ! xdv:      the design variable
  ! qval:     the penalty parameter
  ! qxval:    the RAMP penalty for the material parametrization
  ! Cmat:     the constitutive matrices
  ! ncols:    the length of the columns array
  ! rowp:     the row pointer
  ! cols:     the column index
  !
  ! Output:
  ! K:        the stiffness matrix entries

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: ptype, n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: qval, qxval, Cmat(3,3,nmat)
  integer, intent(in) :: ncols, rowp(n+1), cols(ncols)
  real(kind=dtype), intent(inout) :: K(2,2,ncols)

  ! Temporary data used in the element computation
  integer :: i, ii, jj, jp
  real(kind=dtype) :: Ke(8,8), Celem(3,3)

  ! Constants used in this function
  real(kind=dtype), parameter :: zero = 0.0_dtype
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero all entries in the matrix
  K(:,:,:) = zero

  do i = 1, ne
     ! Compute the element constitutive properties
     call computeConstitutiveMat(i, ptype, ne, ntw, nmat, &
          tconn, tweights, xdv, qval, qxval, Cmat, Celem)

     ! Evaluate the element stiffness matrix
     call computeElemKmat(i, n, ne, conn, X, Celem, Ke)

     ! Add the values into the stiffness matrix
     do ii = 1, 4
        ! Find the columns within the matrix
        do jj = 1, 4
           ! Just do an exhaustive search to find the rows
           do jp = rowp(conn(ii,i)), rowp(conn(ii,i)+1)-1
              if (cols(jp) == conn(jj,i)) then
                 K(:,:,jp) = K(:,:,jp) + Ke(2*ii-1:2*ii, 2*jj-1:2*jj)
                 exit
              end if
           end do
        end do
     end do
  end do

end subroutine computeKmat

subroutine addAmatProduct( &
     ptype, n, ne, ntw, nmat, conn, X, U, &
     tconn, tweights, xdv, xperb, qval, qxval, Cmat, ru)
  ! Compute the global stiffness matrix and store it in the given
  ! compressed sparse row data format.
  !
  ! Input:
  ! ptype:    the type of stiffness parametrization
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the max size of the filter
  ! nmat:     the number of material selections
  ! conn:     the element connectivity
  ! X:        the nodal locations
  ! tconn:    the thickness to design variable weight array
  ! tweights: the thickness weights
  ! xdv:      the design variable
  ! qval:     the penalty parameter
  ! qxval:    the RAMP penalty for the material parametrization
  ! Cmat:     the constitutive matrices
  ! ncols:    the length of the columns array
  ! rowp:     the row pointer
  ! cols:     the column index
  !
  ! Output:
  ! K:        the stiffness matrix entries

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: ptype, n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne)
  real(kind=dtype), intent(in) :: xdv(nmat+1,ne), xperb(nmat+1,ne)
  real(kind=dtype), intent(in) :: qval, qxval, Cmat(3,3,nmat)
  real(kind=dtype), intent(inout) :: ru(2,n)

  ! Temporary data used in the element computation
  integer :: i, ii, jj
  real(kind=dtype) :: Ke(8,8), Celem(3,3)

  do i = 1, ne
     ! Compute the element constitutive properties
     call addConstitutiveMatDerivPerb(i, ptype, ne, ntw, nmat, &
          tconn, tweights, xdv, xperb, qval, qxval, Cmat, Celem)

     ! Evaluate the element stiffness matrix
     call computeElemKmat(i, n, ne, conn, X, Celem, Ke)

     ! Now, compute the residual by multiplying the element matrix
     ! by the perturbation in the design variables
     do ii = 1, 4
        do jj = 1, 4
           ru(:,conn(ii,i)) = ru(:,conn(ii,i)) + &
                matmul(Ke(2*ii-1:2*ii, 2*jj-1:2*jj), U(:,conn(jj,i)))
        end do
     end do
  end do

end subroutine addAmatProduct

subroutine addAmatTransposeProduct( &
     ptype, n, ne, ntw, nmat, conn, X, U, Uperb, &
     tconn, tweights, xdv, qval, qxval, Cmat, rx)
  ! Compute the derivative of the inner product of the vectors U and
  ! Uperb with respect to the design variables.
  !
  ! Input:
  ! ptype:    the type of stiffness parametrization
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the max size of the filter
  ! nmat:     the number of material selections
  ! conn:     the element connectivity
  ! X:        the nodal locations
  ! U:        the nodal displacements
  ! Uperb:    the perturbation to the nodal displacements
  ! tconn:    the thickness to design variable weight array
  ! tweights: the thickness weights
  ! xdv:      the design variable
  ! qval:     the penalty parameter
  ! qxval:    the RAMP penalty for the material parametrization
  ! Cmat:     the constitutive matrices
  !
  ! Output:
  ! rx:       the output product

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: ptype, n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n), Uperb(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: qval, qxval, Cmat(3,3,nmat)
  real(kind=dtype), intent(inout) :: rx(nmat+1,ne)

  ! Temporary data used in the element calculation
  integer :: i, j, k, ii
  real(kind=dtype) :: inner(nmat)
  real(kind=dtype) :: el(3), er(3)
  real(kind=dtype) :: Xd(2,2), Jd(2,2), Ud(2,2)
  real(kind=dtype) :: ns(4), nxi(4), neta(4)
  real(kind=dtype) :: quadpts(2), quadwts(2)
  real(kind=dtype) :: det, invdet, h

  ! Set the Gauss quadrature point/weight values
  quadpts(1) = -0.577350269189626_dtype
  quadpts(2) = 0.577350269189626_dtype
  quadwts(1) = 1.0_dtype
  quadwts(2) = 1.0_dtype

  do i = 1, ne
     inner(:) = 0.0_dtype

     do j = 1, 2
        do k = 1, 2
           ! Evaluate the shape functions
           call evalShapeFunctions(quadpts(j), quadpts(k), ns, nxi, neta)

           ! Evaluate the Jacobian of the residuals
           call getElemGradient(i, n, ne, conn, X, nxi, neta, Xd)

           ! Compute J = Xd^{-1}
           det = Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1)
           invdet = 1.0_dtype/det
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Compute the quadrature weight at this point
           h = quadwts(j)*quadwts(k)*det

           ! Evaluate the left- and right- contributions to the strain
           call getElemGradient(i, n, ne, conn, U, nxi, neta, Ud)
           call evalStrain(Jd, Ud, er)

           call getElemGradient(i, n, ne, conn, Uperb, nxi, neta, Ud)
           call evalStrain(Jd, Ud, el)

           ! Compute the inner product with the constitutive matrix
           do ii = 1, nmat
              inner(ii) = inner(ii) + h*dot_product(el, matmul(Cmat(:,:,ii), er))
           end do
        end do
     end do

     ! Compute the element constitutive properties
     call addConstitutiveMatDeriv(i, ptype, ne, ntw, nmat, &
          tconn, tweights, xdv, qval, qxval, inner, rx)
  end do

end subroutine addAmatTransposeProduct

subroutine computeAllStress( &
     ftype, xi, eta, n, ne, ntw, nmat, &
     conn, X, U, tconn, tweights, xdv, &
     epsilon, h, G, stress)
  ! Compute the stress constraints for each material in all of the
  ! elements in the finite-element mesh.
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! epsilon:  the epsilon relaxation factor
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  !
  ! Output:
  ! stress:   the values of the stress constraints

  use precision
  implicit none

  integer, intent(in) :: ftype
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3,nmat), G(3,3,nmat)
  real(kind=dtype), intent(inout) :: stress(nmat,ne)

  ! Temporary data used internally
  integer :: i, j, k
  real(kind=dtype) :: findex, ttmp, xtmp, e(3), etmp(3)
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero the temp thickness/materials
  ttmp = 0.0_dtype
  xtmp = 0.0_dtype

  ! Evaluate the shape functions at the given point
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  do i = 1, ne
     ! Evaluate the filtered thickness and strain for this element
     ttmp = 0.0_dtype
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Add up the element thickness
           ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))

           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate the stress/strain
           call evalStrain(Jd, Ud, etmp)

           ! Add the result to the local strain
           e = e + tweights(k,i)*etmp
        end if
     end do

     findex = one
     if (ftype == 2) then
        ! Compute the failure index
        findex = one + epsilon/ttmp - epsilon
     end if

     do j = 1, nmat
        ! Evaluate the filtered "thickness" of the material
        xtmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              xtmp = xtmp + tweights(k,i)/xdv(j+1,tconn(k,i))
           end if
        end do

        ! Evaluate the stress constraint from the given material type
        stress(j,i) = findex + epsilon/xtmp - epsilon &
             - (dot_product(e, h(:,j)) + &
             dot_product(e, matmul(G(:,:,j), e)))
     end do
  end do

end subroutine computeAllStress

subroutine computeLogStressSum( &
     ftype, nelems, elems, xi, eta, n, ne, ntw, nmat, &
     conn, X, U, tconn, tweights, xdv, &
     epsilon, h, G, log_sum)
  ! Compute the sum of the log of the stress constraints in all the
  ! elements within the finite-element mesh.
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! nelems:   the number of elements
  ! elems:    the element indices for the stress constraints
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! epsilon:  the epsilon relaxation factor
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  !
  ! Output:
  ! stress:   the values of the stress constraints

  use precision
  implicit none

  integer, intent(in) :: ftype, nelems, elems(nelems)
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3,nmat), G(3,3,nmat)
  real(kind=dtype), intent(out) :: log_sum

  ! Temporary data used internally
  integer :: ie, i, j, k
  real(kind=dtype) :: findex, ttmp, xtmp, e(3), etmp(3), stress
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)
  real(kind=dtype) :: sum_pos, sum_neg

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Initialize the sums
  sum_pos = 0.0_dtype
  sum_neg = 0.0_dtype

  ! Evaluate the shape functions at the central point
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  do ie = 1, nelems
     ! Figure out the element index
     i = elems(ie)

     ! Evaluate the filtered thickness and strain for this element
     ttmp = 0.0_dtype
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Add up the element thickness
           ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))

           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate the stress/strain
           call evalStrain(Jd, Ud, etmp)

           ! Add the result to the local strain
           e = e + tweights(k,i)*etmp
        end if
     end do

     ! Compute the failure index
     findex = one
     if (ftype == 2) then
        findex = one + epsilon/ttmp - epsilon
     end if

     do j = 1, nmat
        ! Evaluate the filtered "thickness" of the material
        xtmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              xtmp = xtmp + tweights(k,i)/xdv(j+1,tconn(k,i))
           end if
        end do

        ! Evaluate the stress constraint from the given material type
        stress = findex + epsilon/xtmp - epsilon &
             - dot_product(e, h(:,j)) - dot_product(e, matmul(G(:,:,j), e))

        if (stress > 1.0) then
           sum_pos = sum_pos + log(stress)
        else
           sum_neg = sum_neg + log(stress)
        end if
     end do
  end do

  log_sum = sum_pos + sum_neg

end subroutine computeLogStressSum

subroutine computeMaxStep( &
     ftype, nelems, elems, xi, eta, n, ne, ntw, nmat, &
     conn, X, U, Ustep, tconn, tweights, xdv, xstep, &
     epsilon, h, G, tau, alpha)
  ! Compute the sum of the log of the stress constraints in all the
  ! elements within the finite-element mesh.
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! nelems:   the number of elements
  ! elems:    the element indices for the stress constraints
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! epsilon:  the epsilon relaxation factor
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! tau:      back-off this fraction from the (approximate) boundary
  ! maxiter:  the maximum number of Newton iterations
  ! tol:      the solution tolerance
  !
  ! Output:
  ! alpha:    the maximum step < 1.0 along the direction U + alpha*Ustep

  use precision
  implicit none

  integer, intent(in) :: ftype, nelems, elems(nelems)
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n), Ustep(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne), xstep(nmat+1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3,nmat), G(3,3,nmat), tau
  real(kind=dtype), intent(out) :: alpha

  ! Temporary data used internally
  integer :: ie, i, j, k, l
  real(kind=dtype) :: findex, dfindex, ddfindex
  real(kind=dtype) :: e(3), es(3), etmp(3)
  real(kind=dtype) :: ttmp, xtmp, dttmp, dxtmp, d2, d3
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)
  real(kind=dtype) :: a, b, c, discrim, r, r1, r2
  real(kind=dtype) :: p(ntw)

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Set the maximum step length
  alpha = 1.0_dtype

  ! Evaluate the shape functions at the given point
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  do ie = 1, nelems
     ! Retrieve the element index
     i = elems(ie)

     ! Compute the failure index
     findex = 1.0_dtype
     dfindex = 0.0_dtype
     ddfindex = 0.0_dtype

     if (ftype == 2) then
        ! Evaluate the filtered thickness of the given element
        ttmp = 0.0_dtype
        dttmp = 0.0_dtype
        d3 = 0.0_dtype
        p(:) = 0.0_dtype

        ! Compute the derivatives along the given direction
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))
              p(k) = xstep(1,tconn(k,i))*tweights(k,i)/xdv(1,tconn(k,i))**2
              dttmp = dttmp - p(k)
              d3 = d3 + tweights(k,i)*xstep(1,tconn(k,i))**2/xdv(1,tconn(k,i))**3
           end if
        end do

        d2 = 0.0_dtype
        do k = 1, ntw
           do l = 1, ntw
              d2 = d2 + p(k)*p(l)
           end do
        end do

        ! Compute the failure index
        findex = one + epsilon/ttmp - epsilon
        dfindex = -epsilon*dttmp/ttmp**2
        ddfindex = 2.0*epsilon*(d2 - d3*ttmp)/ttmp**3
     end if

     ! Evaluate the filtered strain
     e(:) = 0.0_dtype
     es(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evaluate the shape functions and the strain
           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate the stress/strain
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)
           call evalStrain(Jd, Ud, etmp)
           e = e + tweights(k,i)*etmp

           ! Evaluate the stress/strain
           call getElemGradient(tconn(k,i), n, ne, conn, Ustep, nxi, neta, Ud)
           call evalStrain(Jd, Ud, etmp)
           es = es + tweights(k,i)*etmp
        end if
     end do

     do j = 1, nmat
        ! Evaluate the filtered "thickness" of the material
        xtmp = 0.0_dtype
        dxtmp = 0.0_dtype
        d3 = 0.0_dtype
        p(:) = 0.0_dtype

        ! Compute the directional derivatives
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              xtmp = xtmp + tweights(k,i)/xdv(j+1,tconn(k,i))
              p(k) = xstep(1+j,tconn(k,i))*tweights(k,i)/xdv(1+j,tconn(k,i))**2
              dxtmp = dxtmp - p(k)
              d3 = d3 + tweights(k,i)*xstep(1+j,tconn(k,i))**2/xdv(1+j,tconn(k,i))**3
           end if
        end do

        d2 = 0.0_dtype
        do k = 1, ntw
           do l = 1, ntw
              d2 = d2 + p(k)*p(l)
           end do
        end do

        ! Compute the values of the indices
        a = 0.5*ddfindex + epsilon*(d2 - d3*xtmp)/xtmp**3
        b = dfindex - epsilon*dxtmp/xtmp**2
        c = findex + epsilon/xtmp - epsilon

        ! Solve a quadratic equation to determine a good initial
        ! estimate for the max permissible step length
        a = a - dot_product(es, matmul(G(:,:,j), es))
        b = b - dot_product(es, h(:,j)) - 2.0*dot_product(es, matmul(G(:,:,j), e))
        c = c - dot_product(e, h(:,j)) - dot_product(e, matmul(G(:,:,j), e))

        ! Due to the defn of G, a > 0.0
        ! If we're at a feasible point then: c < 0.0
        ! As a result, sqrt(b**2 - 4*a*c) > b
        discrim = sqrt(b**2 - 4.0*a*c)
        if (b > 0.0) then
           r1 = -(b + discrim)/(2.0*a)
           r2 = c/(a*r1)
        else
           r1 = -(b - discrim)/(2.0*a)
           r2 = c/(a*r1)
        end if

        ! Pick the smallest positive root
        if (r1 > 0.0 .and. r2 > 0.0) then
           r = min(r1, r2)
        else
           r = max(r1, r2)
        end if

        alpha = min(tau*r, alpha)
     end do
  end do

end subroutine computeMaxStep

subroutine addLogStressSumDeriv( &
     ftype, nelems, elems, xi, eta, n, ne, ntw, nmat, &
     conn, X, U, tconn, tweights, xdv, &
     epsilon, h, G, barrier, rx, ru)
  ! Add the derivative of the barrier term from the stress constraints
  ! to the vectors rx and ru
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! nelems:   the number of elements
  ! elems:    the element indices for the stress constraints
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! epsilon:  the epsilon relaxation factor
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! barrier:  the barrier value
  !
  ! Output:
  ! rx:       the log-inv-det derivative w.r.t. x
  ! ru:       the log-inv-det derivative w.r.t. u

  use precision
  implicit none

  integer, intent(in) :: ftype, nelems, elems(nelems)
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3,nmat), G(3,3,nmat), barrier
  real(kind=dtype), intent(inout) :: rx(nmat+1,ne), ru(2,n)

  ! Temporary data used internally
  integer :: ie, i, j, k, ii, kk
  real(kind=dtype) :: findex, ttmp, xtmp, fact, stress, e(3), etmp(3)
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), B(3,8)

  ! Set the value of unity!
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Set the xtmp variable to zero
  ttmp = 0.0_dtype
  xtmp = 0.0_dtype

  ! Evaluate the shape functions
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  do ie = 1, nelems
     i = elems(ie)

     ! Evaluate the filtered thickness of the given element
     ttmp = 0.0_dtype
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))

           ! Evaluate the shape functions and the strain
           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate the stress/strain
           call evalStrain(Jd, Ud, etmp)
           e = e + tweights(k,i)*etmp
        end if
     end do

     ! Compute the failure index
     findex = 1.0_dtype
     if (ftype == 2) then
        findex = one + epsilon/ttmp - epsilon
     end if

     do j = 1, nmat
        xtmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              xtmp = xtmp + tweights(k,i)/xdv(j+1,tconn(k,i))
           end if
        end do

        ! Evaluate the stress constraint from the given material type
        stress = findex + epsilon/xtmp - epsilon &
             - dot_product(e, h(:,j)) - dot_product(e, matmul(G(:,:,j), e))

        ! Compute the factor for both the derivatives of the stress
        fact = barrier/stress

        ! Add the design-dependent terms to barrier
        ! First, add the terms from the thickness, if required
        if (ftype == 2) then
           do k = 1, ntw
              if (tconn(k,i) > 0) then
                 rx(1,tconn(k,i)) = rx(1,tconn(k,i)) - &
                      fact*epsilon*tweights(k,i)/(ttmp*xdv(1,tconn(k,i)))**2
              end if
           end do
        end if

        ! Next, add the terms from the materials
        do k = 1, ntw
           ii = tconn(k,i)
           if (ii > 0) then
              rx(1+j,ii) = rx(1+j,ii) - &
                   fact*epsilon*tweights(k,i)/(xtmp*xdv(1+j,tconn(k,i)))**2

              ! Evaluate the derivative of the strain matrix
              call getElemGradient(ii, n, ne, conn, X, nxi, neta, Xd)

              ! Compute the inverse of Xd
              invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
              Jd(1,1) =  invdet*Xd(2,2)
              Jd(2,1) = -invdet*Xd(2,1)
              Jd(1,2) = -invdet*Xd(1,2)
              Jd(2,2) =  invdet*Xd(1,1)

              ! Evaluate the B matrix
              call evalBmat(Jd, nxi, neta, B)

              ! Add the terms to the derivative of the stress
              do kk = 1, 4
                 ru(1,conn(kk,ii)) = ru(1,conn(kk,ii)) + &
                      fact*tweights(k,i)*(dot_product(h(:,j), B(:,2*kk-1)) &
                      + 2.0*dot_product(e, matmul(G(:,:,j), B(:,2*kk-1))))

                 ru(2,conn(kk,ii)) = ru(2,conn(kk,ii)) + &
                      fact*tweights(k,i)*(dot_product(h(:,j), B(:,2*kk)) &
                      + 2.0*dot_product(e, matmul(G(:,:,j), B(:,2*kk))))
              end do
           end if
        end do
     end do
  end do

end subroutine addLogStressSumDeriv

subroutine computeCmatProduct( &
     ftype, nelems, elems, xi, eta, n, ne, ntw, nmat, &
     conn, X, U, tconn, tweights, xdv, &
     epsilon, h, G, pu, ru)
  ! Compute the product of the second derivative of the sum of log
  ! barrier terms with the input vector pu.
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! nelems:   the number of elements with stress constraints
  ! elems:    the element indices with stress constraints
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! epsilon:  the epsilon relaxation factor
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! pu:       the input vector
  !
  ! Output:
  ! ru:       the output vector

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: ftype, nelems, elems(nelems)
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3,nmat), G(3,3,nmat)
  real(kind=dtype), intent(in) :: pu(2,n)
  real(kind=dtype), intent(inout) :: ru(2,n)

  ! Temporary data used in the element computation
  integer :: i, ie, j, k, ii, jj
  real(kind=dtype) :: findex, tmp(nmat+1)
  real(kind=dtype) :: e(3), es(3), etmp(3), stress(nmat), fact
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)
  real(kind=dtype) :: hs, hj(3), B(3,8)

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero the result
  ru(:,:) = 0.0_dtype

  ! Evaluate the shape functions and the strain
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  do ie = 1, nelems
     ! Retrieve the element index
     i = elems(ie)

     ! Evaluate the filtered thickness of the given element
     tmp(:) = 0.0_dtype
     e(:) = 0.0_dtype
     es(:) = 0.0_dtype
     do k = 1, ntw
        ii = tconn(k,i)
        if (ii > 0) then
           tmp = tmp + tweights(k,i)/xdv(:,ii)

           ! Evaluate the shape functions and the strain
           call getElemGradient(ii, n, ne, conn, X, nxi, neta, Xd)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate the stress/strain
           call getElemGradient(ii, n, ne, conn, U, nxi, neta, Ud)
           call evalStrain(Jd, Ud, etmp)
           e = e + tweights(k,i)*etmp

           ! Evaluate the strain from the step
           call getElemGradient(ii, n, ne, conn, pu, nxi, neta, Ud)
           call evalStrain(Jd, Ud, etmp)
           es = es + tweights(k,i)*etmp
        end if
     end do

     ! Compute the failure index
     findex = 1.0_dtype
     if (ftype == 2) then
        findex = one + epsilon/tmp(1) - epsilon
     end if

     ! Evaluate the stress constraint for all of the materials
     do j = 1, nmat
        stress(j) = findex + epsilon/tmp(1+j) - epsilon &
             - dot_product(e, h(:,j)) - dot_product(e, matmul(G(:,:,j), e))
     end do

     ! Add the product to the output vector
     do k = 1, ntw
        ii = tconn(k,i)
        if (ii > 0) then
           ! Evaluate the shape functions and the strain
           call getElemGradient(ii, n, ne, conn, X, nxi, neta, Xd)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate the stress/strain
           call evalBmat(Jd, nxi, neta, B)

           ! Loop over each material
           do j = 1, nmat
              ! Compute the factor
              fact = tweights(k,i)/stress(j)**2

              ! Compute the failure-dependent material strain
              hs = dot_product(h(:,j), es) + &
                   2.0*dot_product(e, matmul(G(:,:,j), es))
              hj = h(:,j) + 2.0*matmul(G(:,:,j), e)

              ! Add the result to the ru vector
              do jj = 1, 4
                 ru(1,conn(jj,ii)) = ru(1,conn(jj,ii)) + &
                      fact*(hs*dot_product(B(:,2*jj-1), hj) + &
                      2.0*stress(j)*dot_product(B(:,2*jj-1), matmul(G(:,:,j), es)))

                 ru(2,conn(jj,ii)) = ru(2,conn(jj,ii)) + &
                      fact*(hs*dot_product(B(:,2*jj), hj) + &
                      2.0*stress(j)*dot_product(B(:,2*jj), matmul(G(:,:,j), es)))
              end do
           end do
        end if
     end do
  end do

end subroutine computeCmatProduct

subroutine computeElemCmat( &
     index, ftype, xi, eta, n, ne, ntw, nmat, &
     conn, X, U, tconn, tweights, xdv, epsilon, h, G, Cu)
  ! Compute the second derivative of the barrier term with respect to
  ! the full space. This code computes the block-diagonal components
  ! of the matrix.
  !
  ! Input:
  ! index:    the element index
  ! ftype:    the type of failure parametrization to use
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! epsilon:  the epsilon relaxation factor
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  !
  ! Output:
  ! Cu:       the second derivative of the barrier term w.r.t. u

  use precision
  implicit none

  integer, intent(in) :: index, ftype
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3,nmat), G(3,3,nmat)
  real(kind=dtype), intent(inout) :: Cu(8,8)

  ! Temporary data used internally
  integer :: i, j, k
  real(kind=dtype) :: findex, ttmp, xtmp, e(3), stress, fact
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)
  real(kind=dtype) :: hi(3), B(3,8)

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero the elements in the ce matrix
  Cu(:,:) = 0.0_dtype

  ! Evaluate the shape functions and the strain
  call evalShapeFunctions(xi, eta, ns, nxi, neta)
  call getElemGradient(index, n, ne, conn, X, nxi, neta, Xd)
  call getElemGradient(index, n, ne, conn, U, nxi, neta, Ud)

  ! Compute the inverse of Xd
  invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
  Jd(1,1) =  invdet*Xd(2,2)
  Jd(2,1) = -invdet*Xd(2,1)
  Jd(1,2) = -invdet*Xd(1,2)
  Jd(2,2) =  invdet*Xd(1,1)

  ! Evaluate the stress/strain
  call evalStrain(Jd, Ud, e)
  call evalBmat(Jd, nxi, neta, B)

  ! Compute the failuter criterion
  findex = 1.0_dtype
  if (ftype == 2) then
     ! Evaluate the filtered thickness of the given element
     ttmp = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,index) > 0) then
           ttmp = ttmp + tweights(k,index)/xdv(1,tconn(k,index))
        end if
     end do

     ! Compute the failure index
     findex = one + epsilon/ttmp - epsilon
  end if

  ! Evaluate the filtered "thickness" of the material
  do i = 1, nmat
     xtmp = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,index) > 0) then
           xtmp = xtmp + tweights(k,index)/xdv(i+1,tconn(k,index))
        end if
     end do

     ! Compute the stress factor
     stress = findex + epsilon/xtmp - epsilon &
          - dot_product(e, h(:,i)) - dot_product(e, matmul(G(:,:,i), e))

     ! Compute the factor
     fact = one/stress**2

     ! Compute the failure-dependent material strain
     hi = h(:,i) + 2.0*matmul(G(:,:,i), e)

     ! Compute the second derivatives w.r.t. u
     do j = 1, 8
        do k = 1, 8
           Cu(j,k) = Cu(j,k) + &
                fact*(dot_product(B(:,j), hi)*dot_product(B(:,k), hi) + &
                2.0*stress*dot_product(B(:,j), matmul(G(:,:,i), B(:,k))))
        end do
     end do
  end do

end subroutine computeElemCmat

subroutine computeCmat( &
     ftype, nelems, elems, xi, eta, n, ne, ntw, nmat, &
     conn, X, U, tconn, tweights, xdv, &
     epsilon, h, G, nccols, crowp, ccols, C)
  ! Compute the second derivatives of the sum of log barrier terms.
  ! This matrix is positive definite.
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! epsilon:  the epsilon relaxation factor
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! ncols:    the length of the columns array
  ! rowp:     the row pointer
  ! cols:     the column index
  !
  ! Output:
  ! C:      the matrix of second derivatives w.r.t. u

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: ftype, nelems, elems(nelems)
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3,nmat), G(3,3,nmat)
  integer, intent(in) :: nccols, crowp(n+1), ccols(nccols)
  real(kind=dtype), intent(inout) :: C(2,2,nccols)

  ! Temporary data used in the element computation
  integer :: i, ie, j, k, ii, jj, jp
  real(kind=dtype) :: findex, ttmp, xtmp
  real(kind=dtype) :: e(3), etmp(3), stress, fact
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)
  real(kind=dtype) :: hi(3), B(3,8)
  real(kind=dtype) :: Cu(8,8)

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero all entries in the matrix
  C(:,:,:) = 0.0_dtype

  ! Evaluate the shape functions and the strain
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  do ie = 1, nelems
     ! Retrieve the element index
     i = elems(ie)

     ! Evaluate the filtered thickness of the given element
     ttmp = 0.0_dtype
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))

           ! Evaluate the shape functions and the strain
           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate the stress/strain
           call evalStrain(Jd, Ud, etmp)
           e = e + tweights(k,i)*etmp
        end if
     end do

     ! Compute the failure index
     findex = 1.0_dtype
     if (ftype == 2) then
        findex = one + epsilon/ttmp - epsilon
     end if

     ! Zero the element matrix
     Cu(:,:) = 0.0_dtype

     ! Evaluate the Jacobian
     call getElemGradient(i, n, ne, conn, X, nxi, neta, Xd)

     ! Compute the inverse of Xd
     invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
     Jd(1,1) =  invdet*Xd(2,2)
     Jd(2,1) = -invdet*Xd(2,1)
     Jd(1,2) = -invdet*Xd(1,2)
     Jd(2,2) =  invdet*Xd(1,1)

     ! Evaluate the stress/strain
     call evalBmat(Jd, nxi, neta, B)

     do j = 1, nmat
        xtmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              xtmp = xtmp + tweights(k,i)/xdv(j+1,tconn(k,i))
           end if
        end do

        ! Evaluate the stress constraint from the given material type
        stress = findex + epsilon/xtmp - epsilon &
             - dot_product(e, h(:,j)) - dot_product(e, matmul(G(:,:,j), e))

        ! Compute the factor
        fact = one/stress**2

        ! Compute the failure-dependent material strain
        hi = h(:,j) + 2.0*matmul(G(:,:,j), e)

        ! Compute the second derivatives w.r.t. u
        do ii = 1, 8
           do jj = 1, 8
              Cu(ii,jj) = Cu(ii,jj) + &
                   fact*(dot_product(B(:,ii), hi)*dot_product(B(:,jj), hi) + &
                   2.0*stress*dot_product(B(:,ii), matmul(G(:,:,j), B(:,jj))))
           end do
        end do
     end do

     ! Add values into the C matrix
     do ii = 1, 4
        ! Find the columns within the matrix
        do jj = 1, 4
           ! Just do an exhaustive search to find the rows
           do jp = crowp(conn(ii,i)), crowp(conn(ii,i)+1)-1
              if (ccols(jp) == conn(jj,i)) then
                 C(:,:,jp) = C(:,:,jp) + Cu(2*ii-1:2*ii, 2*jj-1:2*jj)
                 exit
              end if
           end do
        end do
     end do
  end do

end subroutine computeCmat

subroutine computeDmat( &
     ftype, nelems, elems, xi, eta, n, ne, ntw, nmat, &
     conn, X, U, tconn, tweights, xdv, &
     epsilon, h, G, ncols, rowp, cols, D)
  ! Compute the second derivatives of the sum of log barrier terms.
  ! This matrix is positive definite.
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! nmat:     the number of materials
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! epsilon:  the epsilon relaxation factor
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! ncols:    the length of the columns array
  ! rowp:     the row pointer
  ! cols:     the column index
  !
  ! Output:
  ! D:      the matrix of second derivatives w.r.t. x

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: ftype, nelems, elems(nelems)
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3,nmat), G(3,3,nmat)
  integer, intent(in) :: ncols, rowp(ne+1), cols(ncols)
  real(kind=dtype), intent(inout) :: D(1+nmat,1+nmat,ncols)

  ! Temporary data used in the element computation
  integer :: ie, i, j, k, ii, jj, ki, kj, jp
  real(kind=dtype) :: findex, ttmp, xtmp, e(3), etmp(3)
  real(kind=dtype) :: stress, invs, tw
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)

  ! Create a parameter for unity
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero all entries in the matrix
  D(:,:,:) = 0.0_dtype

  ! Zero out the filtered thickness and material
  ttmp = 0.0_dtype
  xtmp = 0.0_dtype

  ! Evaluate the shape functions and the strain
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  ! Loop over all the elements in the mesh
  do ie = 1, nelems
     ! Retrieve the element index
     i = elems(ie)

     ! Evaluate the filtered thickness of the given element
     ttmp = 0.0_dtype
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))

           ! Evaluate the shape functions and the strain
           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate the stress/strain
           call evalStrain(Jd, Ud, etmp)
           e = e + tweights(k,i)*etmp
        end if
     end do

     ! Compute the failure index
     findex = 1.0_dtype
     if (ftype == 2) then
        findex = one + epsilon/ttmp - epsilon
     end if

     do j = 1, nmat
        ! Compute the material "thickness"
        xtmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              xtmp = xtmp + tweights(k,i)/xdv(j+1,tconn(k,i))
           end if
        end do

        ! Evaluate the stress constraint from the given material type
        stress = findex + epsilon/xtmp - epsilon &
             - dot_product(e, h(:,j)) - dot_product(e, matmul(G(:,:,j), e))

        ! Compute the inverse of the stress
        invs = one/stress**2

        do ii = 1, ntw
           ki = tconn(ii,i)
           if (ki > 0) then
              do jj = 1, ntw
                 kj = tconn(jj,i)
                 if (kj > 0) then
                    ! Add the values of the derivatives
                    tw = epsilon*tweights(ii,i)*tweights(jj,i)*invs

                    ! Just do an exhaustive search to find the entry
                    do jp = rowp(ki), rowp(ki+1)-1
                       if (cols(jp) == kj) then
                          ! If the thickness variable is also a
                          ! topology variable
                          if (ftype == 2) then
                             D(1,1,jp) = D(1,1,jp) + &
                                  tw*(epsilon - 2.0*stress*ttmp)/ &
                                  (xdv(1,ki)*xdv(1,kj)*ttmp**2)**2

                             ! Add the contributions to the cross-terms
                             D(1,1+j,jp) = D(1,1+j,jp) + &
                                  tw*epsilon/(xdv(1,ki)*xdv(1+j,kj)*xtmp*ttmp)**2
                             D(1+j,1,jp) = D(1+j,1,jp) + &
                                  tw*epsilon/(xdv(1+j,ki)*xdv(1,kj)*xtmp*ttmp)**2

                             ! If ii == jj, add the additional second
                             ! derivative contribution
                             if (ii == jj) then
                                D(1,1,jp) = D(1,1,jp) + &
                                     2.0*epsilon*tweights(ii,i)/ &
                                     (stress*(ttmp**2)*(xdv(1,ki)**3))
                             end if
                          end if

                          ! Add the terms from the material selection
                          ! variable
                          D(1+j,1+j,jp) = D(1+j,1+j,jp) + &
                               tw*(epsilon - 2.0*stress*xtmp)/ &
                               (xdv(1+j,ki)*xdv(1+j,kj)*xtmp**2)**2

                          ! If ii == jj, add the additional second
                          ! derivative contribution
                          if (ii == jj) then
                             D(1+j,1+j,jp) = D(1+j,1+j,jp) + &
                                  2.0*epsilon*tweights(ii,i)/ &
                                  (stress*(xtmp**2)*(xdv(1+j,ki)**3))
                          end if

                          ! Quit the loop, we're done here
                          exit
                       end if
                    end do
                 end if
              end do
           end if
        end do
     end do
  end do

end subroutine computeDmat

subroutine addDmatIndef( &
     exact, ptype, n, ne, ntw, nmat, conn, X, U, psi, &
     tconn, tweights, xdv, qval, qxval, Cmat, ncols, rowp, cols, D)
  ! Compute the second derivative of the inner product of the
  ! state variables U, and the multiplier vector psi with
  ! the stiffness matrix:
  !
  ! Dmat = Dmat + d/dx^2(U^{T}*K(x)*psi)
  !
  ! Input:
  ! ptype:    the type of stiffness parametrization
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the max size of the filter
  ! nmat:     the number of material selections
  ! conn:     the element connectivity
  ! X:        the nodal locations
  ! tconn:    the thickness to design variable weight array
  ! tweights: the thickness weights
  ! xdv:      the design variable
  ! qval:     the penalty parameter
  ! qxval:    the RAMP penalty for the material parametrization
  ! Cmat:     the constitutive matrices
  ! ncols:    the length of the columns array
  ! rowp:     the row pointer
  ! cols:     the column index
  !
  ! Output:
  ! D:        the new derivative entries

  use precision
  implicit none

  ! The input data
  logical, intent(in) :: exact
  integer, intent(in) :: ptype, n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n), psi(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne)
  real(kind=dtype), intent(in) :: xdv(nmat+1,ne)
  real(kind=dtype), intent(in) :: qval, qxval, Cmat(3,3,nmat)
  integer, intent(in) :: ncols, rowp(ne+1), cols(ncols)
  real(kind=dtype), intent(inout) :: D(1+nmat,1+nmat,ncols)

  ! Temporary data used in the element computation
  integer :: i, j, k, ii, jj, ki, kj, jp
  real(kind=dtype) :: Ke(8,8), ue(8), psie(8), inner(nmat), sum
  real(kind=dtype) :: tmp(nmat+1), ptmp(nmat+1)
  real(kind=dtype) :: dptmp(nmat+1), d2ptmp(nmat+1), tw

  do i = 1, ne
     ! Get the element displacement and multipliers
     do j = 1, 4
        ue(2*j-1) = U(1,conn(j,i))
        ue(2*j) = U(2,conn(j,i))
        psie(2*j-1) = psi(1,conn(j,i))
        psie(2*j) = psi(2,conn(j,i))
     end do

     ! Compute the inner products
     inner(:) = 0.0_dtype
     do j = 1, nmat
        ! Evaluate the element stiffness matrix
        call computeElemKmat(i, n, ne, conn, X, Cmat(:,:,j), Ke)
        inner(j) = dot_product(ue, matmul(Ke, psie))
        if ((exact .eqv. .false.) .and. inner(j) < 0.0) then
           inner(j) = 0.0_dtype
        end if
     end do

     ! Evaluate the filtered values of the variables
     tmp(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           tmp = tmp + tweights(k,i)/xdv(:,tconn(k,i))
        end if
     end do

     ! Compute the penalization
     ptmp(1) = tmp(1)
     dptmp(1) = 1.0_dtype
     d2ptmp(1) = 0.0_dtype
     if (ptype == 2) then
        call computePenalty2ndDeriv(tmp(1), qval, ptmp(1), dptmp(1), d2ptmp(1))
     end if
     do j = 2, nmat+1
        call computePenalty2ndDeriv(tmp(j), qxval, ptmp(j), dptmp(j), d2ptmp(j))
     end do

     sum = 0.0_dtype
     do j = 1, nmat
        sum = sum + ptmp(j+1)*inner(j)
     end do

     ! Add the terms to the second derivative matrix
     do ii = 1, ntw
        ki = tconn(ii,i)
        if (ki > 0) then
           do jj = 1, ntw
              kj = tconn(jj,i)
              if (kj > 0) then
                 ! Add the values of the
                 do jp = rowp(ki), rowp(ki+1)-1
                    if (cols(jp) == kj) then
                       tw = tweights(ii,i)*tweights(jj,i)

                       ! Add the result from the derivative of the thickness
                       D(1,1,jp) = D(1,1,jp) + &
                            sum*d2ptmp(1)*tw/(xdv(1,ki)*xdv(1,kj))**2

                       do j = 1, nmat
                          ! Add terms from the product of the
                          ! thickness and material-selection terms
                          D(1,1+j,jp) = D(1,1+j,jp) + &
                               inner(j)*dptmp(1)*dptmp(1+j)*tw/(xdv(1,ki)*xdv(1+j,kj))**2

                          D(1+j,1,jp) = D(1+j,1,jp) + &
                               inner(j)*dptmp(1)*dptmp(1+j)*tw/(xdv(1+j,ki)*xdv(1,kj))**2

                          ! Add the terms from the second derivatives
                          ! of the material-selection variables
                          D(1+j,1+j,jp) = D(1+j,1+j,jp) + &
                               ptmp(1)*inner(j)*d2ptmp(1+j)*tw/(xdv(1+j,ki)*xdv(1+j,kj))**2
                       end do

                       ! Add the second derivative terms
                       if (ii == jj) then
                          D(1,1,jp) = D(1,1,jp) + &
                               2.0*sum*dptmp(1)*tweights(ii,i)/xdv(1,ki)**3

                          do j = 1, nmat
                             D(1+j,1+j,jp) = D(1+j,1+j,jp) + &
                                  2.0*ptmp(1)*inner(j)*dptmp(1+j)*tweights(ii,i)/xdv(1+j,ki)**3
                          end do
                       end if

                       exit
                    end if
                 end do
              end if
           end do
        end if
     end do
  end do

end subroutine addDmatIndef

subroutine addCmatOffDiagProduct( &
     ftype, nelems, elems, xi, eta, n, ne, ntw, nmat, &
     conn, X, U, Ustep, tconn, tweights, xdv, xstep, &
     epsilon, h, G, barrier, rx, ru)
  ! Compute the off-diagonal matrix-vector product with the second
  ! derivatives of the stress-barrier term with respect to the design
  ! and state variables. These constitute the off-diagonal terms
  ! within the matrix of second derivatives.
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! xi, eta:  the parametric location within the element
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the number of elements within the filter
  ! nmat:     the number of material selection variables
  ! conn:     the element connectivity
  ! X:        the nodal locations
  ! U:        the displacements at the nodes
  ! Ustep:    the step along the state-variable direction
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! xdv:      the values of the design variables
  ! xtep:     the step along the design variable direction
  ! epsilon:  the epsilon relaxation factor
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  !
  ! Output:
  ! rx:       the product w.r.t. design variables
  ! ru:       the product w.r.t. the state variables

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: ftype, nelems, elems(nelems)
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, nmat, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n), Ustep(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne)
  real(kind=dtype), intent(in) :: xdv(nmat+1,ne), xstep(nmat+1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3,nmat), G(3,3,nmat)
  real(kind=dtype), intent(in) :: barrier
  real(kind=dtype), intent(inout) :: rx(nmat+1,ne), ru(2,n)

  ! Temporary data used in the element computation
  integer :: ie, i, j, k, ii, kk
  real(kind=dtype) :: findex, ttmp, xtmp, stress
  real(kind=dtype) :: dttmp, dxtmp, invs
  real(kind=dtype) :: e(3), es(3), etmp(3)
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), B(3,8)

  ! Create a parameter for unity
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero out the filtered thickness and material
  ttmp = 0.0_dtype
  xtmp = 0.0_dtype
  dttmp = 0.0_dtype
  dxtmp = 0.0_dtype

  ! Evaluate the shape functions
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  ! Loop over all the elements in the mesh
  do ie = 1, nelems
     ! Retrieve the element index
     i = elems(ie)

     ! Evaluate the filtered thickness of the given element
     ttmp = 0.0_dtype
     dttmp = 0.0_dtype
     e(:) = 0.0_dtype
     es(:) = 0.0_dtype
     do k = 1, ntw
        ii = tconn(k,i)
        if (ii > 0) then
           ! Evaluate the product of the step and the derivative
           ttmp = ttmp + tweights(k,i)/xdv(1,ii)
           dttmp = dttmp - xstep(1,ii)*tweights(k,i)/xdv(1,ii)**2

           ! Evaluate the shape functions and the strain
           call getElemGradient(ii, n, ne, conn, X, nxi, neta, Xd)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate the strain
           call getElemGradient(ii, n, ne, conn, U, nxi, neta, Ud)
           call evalStrain(Jd, Ud, etmp)
           e = e + tweights(k,i)*etmp

           ! Evaluate the strain due to the step
           call getElemGradient(ii, n, ne, conn, Ustep, nxi, neta, Ud)
           call evalStrain(Jd, Ud, etmp)
           es = es + tweights(k,i)*etmp
        end if
     end do

     ! Compute the failure index
     findex = 1.0_dtype
     if (ftype == 2) then
        findex = one + epsilon/ttmp - epsilon
     end if

     do j = 1, nmat
        ! Compute the material "thickness"
        xtmp = 0.0_dtype
        dxtmp = 0.0_dtype
        do k = 1, ntw
           ii = tconn(k,i)
           if (ii > 0) then
              xtmp = xtmp + tweights(k,i)/xdv(j+1,ii)
              dxtmp = dxtmp - xstep(j+1,ii)*tweights(k,i)/xdv(j+1,ii)**2
           end if
        end do

        ! Evaluate the stress constraint from the given material type
        stress = findex + epsilon/xtmp - epsilon &
             - dot_product(e, h(:,j)) - dot_product(e, matmul(G(:,:,j), e))

        ! Add the contributions to the product with the state variables
        invs = - barrier*epsilon*(dot_product(es, h(:,j)) + &
             2.0*dot_product(e, matmul(G(:,:,j), es)))/stress**2

        if (ftype == 2) then
           do k = 1, ntw
              ii = tconn(k,i)
              if (ii > 0) then
                 rx(1,ii) = rx(1,ii) + &
                      invs*tweights(k,i)/(ttmp*xdv(1,ii))**2
              end if
           end do
        end if

        do k = 1, ntw
           ii = tconn(k,i)
           if (ii > 0) then
              rx(1+j,ii) = rx(1+j,ii) + &
                   invs*tweights(k,i)/(xtmp*xdv(1+j,ii))**2
           end if
        end do

        ! Add the terms for the product with the state variables
        invs = barrier*epsilon*dxtmp/(xtmp*stress)**2
        if (ftype == 2) then
           invs = invs + barrier*epsilon*dttmp/(ttmp*stress)**2
        end if

        do k = 1, ntw
           ii = tconn(k,i)
           if (ii > 0) then
              ! Evaluate the shape functions and the strain
              call getElemGradient(ii, n, ne, conn, X, nxi, neta, Xd)

              ! Compute the inverse of Xd
              invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
              Jd(1,1) =  invdet*Xd(2,2)
              Jd(2,1) = -invdet*Xd(2,1)
              Jd(1,2) = -invdet*Xd(1,2)
              Jd(2,2) =  invdet*Xd(1,1)

              ! Evaluate the B-matrix
              call evalBmat(Jd, nxi, neta, B)

              do kk = 1, 4
                 ru(1,conn(kk,ii)) = ru(1,conn(kk,ii)) + &
                      invs*tweights(k,i)*(dot_product(h(:,j), B(:,2*kk-1)) &
                      + 2.0*dot_product(e, matmul(G(:,:,j), B(:,2*kk-1))))

                 ru(2,conn(kk,ii)) = ru(2,conn(kk,ii)) + &
                      invs*tweights(k,i)*(dot_product(h(:,j), B(:,2*kk)) &
                      + 2.0*dot_product(e, matmul(G(:,:,j), B(:,2*kk))))
              end do
           end if
        end do
     end do
  end do

end subroutine addCmatOffDiagProduct

subroutine computeWeightCon(ne, nmat, xdv, c)
  ! Compute the values of the weighting constraints
  !
  ! Input:
  ! ne:    the number of elements
  ! nmat:  the number of materials
  ! xdv:   the design variable values
  !
  ! Output:
  ! c:     the values of the weighting constraints

  use precision
  implicit none

  integer, intent(in) :: ne, nmat
  real(kind=dtype), intent(in) :: xdv(nmat+1,ne)
  real(kind=dtype), intent(inout) :: c(ne)

  ! Temporary internal variables
  integer :: i, j

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  c(:) = 0.0_dtype
  do i = 1, ne
     do j = 1, nmat
        c(i) = c(i) + one/xdv(1+j,i)
     end do
     c(i) = c(i) - 1.0_dtype
  end do

end subroutine computeWeightCon

subroutine multAwMatAdd(ne, nmat, xdv, xstep, c)
  ! Add the matrix-vector product of the Jacobian of the weighting
  ! constraints to the output vector
  !
  ! Input:
  ! ne:    the number of elements
  ! nmat:  the number of materials
  ! xdv:   the design variable values
  ! xstep: the input vector for the Jacobian computation
  !
  ! Output:
  ! c:     the values of the weighting constraints

  use precision
  implicit none

  integer, intent(in) :: ne, nmat
  real(kind=dtype), intent(in) :: xdv(nmat+1,ne), xstep(nmat+1,ne)
  real(kind=dtype), intent(inout) :: c(ne)

  ! Temporary internal variables
  integer :: i, j

  do i = 1, ne
     do j = 1, nmat
        c(i) = c(i) - xstep(1+j,i)/xdv(1+j,i)**2
     end do
  end do

end subroutine multAwMatAdd

subroutine multAwMatTransposeAdd(ne, nmat, xdv, cstep, xout)
  ! Add the matrix-vector product of the Jacobian of the weighting
  ! constraints to the output vector
  !
  ! Input:
  ! ne:    the number of elements
  ! nmat:  the number of materials
  ! xdv:   the design variable values
  ! xstep: the input vector for the Jacobian computation
  !
  ! Output:
  ! c:     the values of the weighting constraints

  use precision
  implicit none

  integer, intent(in) :: ne, nmat
  real(kind=dtype), intent(in) :: xdv(nmat+1,ne), cstep(ne)
  real(kind=dtype), intent(inout) :: xout(nmat+1,ne)

  ! Temporary internal variables
  integer :: i, j

  do i = 1, ne
     do j = 1, nmat
        xout(1+j,i) = xout(1+j,i) - cstep(i)/xdv(1+j,i)**2
     end do
  end do

end subroutine multAwMatTransposeAdd

subroutine computeNullSpace(ne, nmat, xdv, Adiag, Z, Aw)
  ! Compute the null-space matrices, stored in Z, Y and the diagonal
  ! matrix given by Aw*Aw^{T} required for the computation of the
  ! step.
  !
  ! Input:
  ! ne:     the number of elements
  ! nmat:   the number of materials
  ! xdv:    the design variable values
  !
  ! Output:
  ! Adiag:  the diagonal matrix = Aw*Aw^{T}
  ! Z:      the null-space basis
  ! Aw:     the constraint matrix

  use precision
  implicit none

  integer, intent(in) :: ne, nmat
  real(kind=dtype), intent(in) :: xdv(nmat+1,ne)
  real(kind=dtype), intent(inout) :: Adiag(ne), Z(nmat+1,nmat,ne)
  real(kind=dtype), intent(inout) :: Aw(nmat+1,ne)

  ! Temporary internal variables
  integer :: i, j, jmin
  real(kind=dtype) :: minval

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero all of the input variables
  Adiag(:) = 0.0_dtype
  Z(:,:,:) = 0.0_dtype
  Aw(:,:) = 0.0_dtype

  do i = 1, ne
     jmin = 1
     minval = xdv(2,i)
     do j = 1, nmat
        ! Compute the product
        Aw(1+j,i) = -one/xdv(1+j,i)**2
        Adiag(i) = Adiag(i) + Aw(1+j,i)**2

        ! Check if this is indeed the minimum value
        if (xdv(1+j,i) < minval) then
           minval = xdv(1+j,i)
           jmin = j
        end if
     end do

     Z(1,1,i) = one

     do j = 1, jmin-1
        Z(1+j,1+j,i) = one
        Z(1+jmin,1+j,i) = -Aw(1+j,i)/Aw(1+jmin,i)
     end do

     do j = jmin+1, nmat
        Z(1+j,j,i) = one
        Z(1+jmin,j,i) = -Aw(1+j,i)/Aw(1+jmin,i)
     end do
  end do

end subroutine computeNullSpace

subroutine computeNullRHS(ne, nmat, Adiag, Z, Aw, &
     ncols, rowp, cols, D, rx, rw, rhs)
  ! Compute the right-hand-side:
  !
  ! Z^{T}*(rx - D*Aw^{T}*Adiag^{-1}*rw)
  !
  ! Input:
  ! ne:     the number of elements
  ! nmat:   the number of materials
  ! xdv:    the design variable values
  ! Adiag:  the diagonal matrix = Aw*Aw^{T}
  ! Z:      the null-space basis
  ! Aw:     the constraint matrix
  ! ncols:  the number of entries in the D matrix
  ! rowp:   the row pointer into the matrix
  ! cols:   the column indices
  ! D:      the entries in the D matrix
  ! rx:     the residual in the design variables
  ! rw:     the constraint values
  !
  ! Output:
  ! rhs:    the right-hand-side

  use precision
  implicit none

  integer, intent(in) :: ne, nmat
  real(kind=dtype), intent(in) :: Adiag(ne), Z(nmat+1,nmat,ne)
  real(kind=dtype), intent(in) :: Aw(nmat+1,ne)
  integer, intent(in) :: ncols, rowp(ne+1), cols(ncols)
  real(kind=dtype), intent(in) :: D(nmat+1,nmat+1,ncols)
  real(kind=dtype), intent(in) :: rx(nmat+1,ne), rw(ne)
  real(kind=dtype), intent(inout) :: rhs(nmat,ne)

  ! Internal temporary data
  integer :: i, j, jp
  real(kind=dtype) :: t(nmat+1)

  ! Multiply the D-matrix
  do i = 1, ne
     t(:) = rx(:,i)
     do jp = rowp(i), rowp(i+1)-1
        j = cols(jp)
        t(:) = t(:) - (rw(j)/Adiag(j))*matmul(D(:,:,jp), Aw(:,j))
     end do

     rhs(:,i) = matmul(transpose(Z(:,:,i)), t)
  end do

end subroutine computeNullRHS

subroutine computeNullSolution(ne, nmat, Adiag, Z, Aw, zup, rw, px)
  ! Given zup = [Z^{T}*D*Z]^{-1}*zrhs, compute:
  !
  ! px = Z*zup + Aw^{T}*Adiag^{-1}*c
  !
  ! Input:
  ! ne:     the number of elements
  ! nmat:   the number of materials
  ! xdv:    the design variable values
  ! Adiag:  the diagonal matrix = Aw*Aw^{T}
  ! Z:      the null-space basis
  ! Aw:     the constraint matrix
  ! ncols:  the number of entries in the D matrix
  ! rowp:   the row pointer into the matrix
  ! cols:   the column indices
  ! D:      the entries in the D matrix
  ! rx:     the residual in the design variables
  ! rw:     the constraint values
  !
  ! Output:
  ! rhs:    the right-hand-side

  use precision
  implicit none

  integer, intent(in) :: ne, nmat
  real(kind=dtype), intent(in) :: Adiag(ne), Z(nmat+1,nmat,ne)
  real(kind=dtype), intent(in) :: Aw(nmat+1,ne)
  real(kind=dtype), intent(in) :: zup(nmat,ne), rw(ne)
  real(kind=dtype), intent(inout) :: px(nmat+1,ne)

  ! Internal temporary data
  integer :: i

  ! Multiply the D-matrix
  do i = 1, ne
     px(:,i) = matmul(Z(:,:,i), zup(:,i)) + Aw(:,i)*(rw(i)/Adiag(i))
  end do

end subroutine computeNullSolution
