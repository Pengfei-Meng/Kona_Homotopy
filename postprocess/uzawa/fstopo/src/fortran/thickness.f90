! The following code implements a full space method for
! stress-constrained topology optimization. These subroutines should
! be called from python once they are wrapped with f2py.
!
! Note: The ptype and ftype denote the type of design parametrization
! and failure formulation, respectively. These values mean the
! following:
!
! ptype =
! 1) Inverse thickness variables
! 2) Inverse thickness variables with RAMP/SIMP penalization
! 3) Direct thickness variables
! 4) Direct thickness variables with RAMP/SIMP penalization
!
! ftype =
! 1) Direct stress constraints (appropriate for thickness optimization)
! 2) Epsilon-relaxation constraints (required for topology optimization)
!
! Copyright (c) Graeme J. Kennedy 2015, All rights reserved.

subroutine computeConstitutiveMat(index, ptype, ne, ntw, &
     tconn, tweights, xdv, qval, Cmat, Celem)
  ! Compute the constitutive matrix for the given element index using
  ! the specified parameterization - either a continuous thickness or
  ! a RAMP interpolation.
  !
  ! Input:
  ! index:    the element index
  ! ptype:    the type of stiffness parametrization
  ! ne:       the number of elements
  ! ntw:      the filter size
  ! tconn:    the connectivity of the filter
  ! tweights: the filter weights
  ! xdv:      the design variables
  ! qval:     the RAMP penalty value
  ! Cmat:     the material selection variables
  !
  ! Output:
  ! Celem:    the element constitutive object

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: index, ptype, ne, ntw, tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: qval, Cmat(3,3)
  real(kind=dtype), intent(inout) :: Celem(3,3)

  ! Temporary data used in this function
  integer :: j
  real(kind=dtype) :: ttmp, tpenalty

  ! Set the parameters for this function
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Compute the filtered thickness
  ttmp = 0.0_dtype
  if (ptype == 3 .or. ptype == 4) then
     do j = 1, ntw
        if (tconn(j,index) > 0) then
           ttmp = ttmp + tweights(j,index)*xdv(1,tconn(j,index))
        end if
     end do
  else
     do j = 1, ntw
        if (tconn(j,index) > 0) then
           ttmp = ttmp + tweights(j,index)/xdv(1,tconn(j,index))
        end if
     end do
  end if

  ! Compute the thickness penalty factor
  tpenalty = ttmp
  if (ptype == 2 .or. ptype == 4) then
     call computePenalty(ttmp, qval, tpenalty)
  end if

  Celem(:,:) = tpenalty*Cmat(:,:)

end subroutine computeConstitutiveMat

subroutine addConstitutiveMatDeriv(index, ptype, ne, ntw, &
     tconn, tweights, xdv, qval, Cmat, el, er, alpha, rx)
  ! Add the inner product of the constitutive matrix with a given
  ! vectors to the given vector. Note that the derivative of this
  ! vector will depend on the
  !
  ! Input:
  ! index:    the element index
  ! ptype:    the type of stiffness parametrization
  ! ne:       the number of elements
  ! ntw:      the filter size
  ! tconn:    the connectivity of the filter
  ! tweights: the filter weights
  ! xdv:      the design variables
  ! qval:     the RAMP penalty value
  ! Cmat:     the material selection variables
  ! el:       the strain from the left-hand-side
  ! er:       the strain from the right-hand-side
  ! alpha:    the scalar to use
  !
  ! Output:
  ! rx:       the derivative

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: index, ptype, ne, ntw, tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne), qval
  real(kind=dtype), intent(in) :: Cmat(3,3), el(3), er(3), alpha
  real(kind=dtype), intent(inout) :: rx(1,ne)

  ! Temporary data used in this function
  integer :: j, k
  real(kind=dtype) :: ttmp, tpenalty
  real(kind=dtype) :: tderiv, inner

  ! Set the parameters for this function
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Compute the filtered thickness
  ttmp = 0.0_dtype
  if (ptype == 3 .or. ptype == 4) then
     do j = 1, ntw
        if (tconn(j,index) > 0) then
           ttmp = ttmp + tweights(j,index)*xdv(1,tconn(j,index))
        end if
     end do
  else
     do j = 1, ntw
        if (tconn(j,index) > 0) then
           ttmp = ttmp + tweights(j,index)/xdv(1,tconn(j,index))
        end if
     end do
  end if

  ! Compute the thickness penalty factor
  tpenalty = ttmp
  tderiv = 1.0_dtype
  if (ptype == 2 .or. ptype == 4) then
     call computePenaltyDeriv(ttmp, qval, tpenalty, tderiv)
  end if

  ! Compute the inner product with the constitutive matrix
  inner = dot_product(el, matmul(Cmat, er))

  ! Add up the contributions from each donnor cell
  if (ptype == 3 .or. ptype == 4) then
     do j = 1, ntw
        k = tconn(j,index)
        if (k > 0) then
           ! Compute the contribution to the thickness parameter
           rx(1,k) = rx(1,k) &
                + alpha*inner*tderiv*tweights(j,index)
        end if
     end do
  else
     do j = 1, ntw
        k = tconn(j,index)
        if (k > 0) then
           ! Compute the contribution to the thickness parameter
           rx(1,k) = rx(1,k) &
                - alpha*inner*tderiv*tweights(j,index)/xdv(1,k)**2
        end if
     end do
  end if

end subroutine addConstitutiveMatDeriv

subroutine addConstitutiveMatDerivPerb( &
     index, ptype, ne, ntw, &
     tconn, tweights, xdv, xperb, qval, Cmat, Celem)
  ! Add the inner product of the constitutive matrix with a given
  ! vectors to the given vector. Note that the derivative of this
  ! vector will depend on the
  !
  ! Input:
  ! index:    the element index
  ! ptype:    the type of stiffness parametrization
  ! ne:       the number of elements
  ! ntw:      the filter size
  ! tconn:    the connectivity of the filter
  ! tweights: the filter weights
  ! xdv:      the design variables
  ! xperb:    the perturbation of the design variables
  ! qval:     the RAMP penalty value
  ! Cmat:     the material selection variables
  !
  ! Output:
  ! Celem:    the element constitutive object

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: index, ptype, ne, ntw, tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: xperb(1,ne), qval
  real(kind=dtype), intent(in) :: Cmat(3,3)
  real(kind=dtype), intent(inout) :: Celem(3,3)

  ! Temporary data used in this function
  integer :: j, kj
  real(kind=dtype) :: ttmp, dttmp
  real(kind=dtype) :: tpenalty, tderiv

  ! Set the parameters for this function
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Compute the filtered thickness
  ttmp = 0.0_dtype
  dttmp = 0.0_dtype
  if (ptype == 3 .or. ptype == 4) then
     do j = 1, ntw
        kj = tconn(j,index)
        if (kj > 0) then
           ttmp = ttmp + tweights(j,index)*xdv(1,kj)
           dttmp = dttmp + xperb(1,kj)*tweights(j,index)
        end if
     end do
  else
     do j = 1, ntw
        kj = tconn(j,index)
        if (kj > 0) then
           ttmp = ttmp + tweights(j,index)/xdv(1,kj)
           dttmp = dttmp - xperb(1,kj)*tweights(j,index)/xdv(1,kj)**2
        end if
     end do
  end if

  ! Compute the thickness penalty factor
  tderiv = dttmp
  if (ptype == 2 .or. ptype == 4) then
     call computePenaltyDeriv(ttmp, qval, tpenalty, tderiv)
     tderiv = dttmp*tderiv
  end if

  Celem(:,:) = tderiv*Cmat(:,:)

end subroutine addConstitutiveMatDerivPerb

subroutine computeKmat( &
     ptype, n, ne, ntw, conn, X, &
     tconn, tweights, xdv, qval, Cmat, &
     ncols, rowp, cols, K)
  ! Compute the global stiffness matrix and store it in the given
  ! compressed sparse row data format.
  !
  ! Input:
  ! ptype:    the type of stiffness parametrization
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the max size of the filter
  ! conn:     the element connectivity
  ! X:        the nodal locations
  ! tconn:    the thickness to design variable weight array
  ! tweights: the thickness weights
  ! xdv:      the design variable
  ! qval:     the penalty parameter
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
  integer, intent(in) :: ptype, n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: qval, Cmat(3,3)
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
     call computeConstitutiveMat(i, ptype, ne, ntw, &
          tconn, tweights, xdv, qval, Cmat, Celem)

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
     ptype, n, ne, ntw, conn, X, U, &
     tconn, tweights, xdv, xperb, qval, Cmat, ru)
  ! Compute the global stiffness matrix and store it in the given
  ! compressed sparse row data format.
  !
  ! Input:
  ! ptype:    the type of stiffness parametrization
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the max size of the filter
  ! conn:     the element connectivity
  ! X:        the nodal locations
  ! tconn:    the thickness to design variable weight array
  ! tweights: the thickness weights
  ! xdv:      the design variable
  ! qval:     the penalty parameter
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
  integer, intent(in) :: ptype, n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne)
  real(kind=dtype), intent(in) :: xdv(1,ne), xperb(1,ne)
  real(kind=dtype), intent(in) :: qval, Cmat(3,3)
  real(kind=dtype), intent(inout) :: ru(2,n)

  ! Temporary data used in the element computation
  integer :: i, ii, jj
  real(kind=dtype) :: Ke(8,8), Celem(3,3)

  do i = 1, ne
     ! Compute the element constitutive properties
     call addConstitutiveMatDerivPerb(i, ptype, ne, ntw, &
          tconn, tweights, xdv, xperb, qval, Cmat, Celem)

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
     ptype, n, ne, ntw, conn, X, U, Uperb, &
     tconn, tweights, xdv, qval, Cmat, rx)
  ! Compute the derivative of the inner product of the vectors U and
  ! Uperb with respect to the design variables.
  !
  ! Input:
  ! ptype:    the type of stiffness parametrization
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the max size of the filter
  ! conn:     the element connectivity
  ! X:        the nodal locations
  ! U:        the nodal displacements
  ! Uperb:    the perturbation to the nodal displacements
  ! tconn:    the thickness to design variable weight array
  ! tweights: the thickness weights
  ! xdv:      the design variable
  ! qval:     the penalty parameter
  ! Cmat:     the constitutive matrices
  !
  ! Output:
  ! rx:       the output product

  use precision
  implicit none

  ! The input data
  integer, intent(in) :: ptype, n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n), Uperb(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: qval, Cmat(3,3)
  real(kind=dtype), intent(inout) :: rx(1,ne)

  ! Temporary data used in the element calculation
  integer :: i, j, k
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
     ! Compute the integral of the
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

           ! Compute the element constitutive properties
           call addConstitutiveMatDeriv(i, ptype, ne, ntw, &
                tconn, tweights, xdv, qval, Cmat, el, er, h, rx)
        end do
     end do
  end do

end subroutine addAmatTransposeProduct

subroutine computeAllStress( &
     ftype, xi, eta, n, ne, ntw, &
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
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3), G(3,3)
  real(kind=dtype), intent(inout) :: stress(ne)

  ! Temporary data used internally
  integer :: i, k
  real(kind=dtype) :: findex, ttmp, e(3)
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero the temp thickness/materials
  ttmp = 0.0_dtype

  do i = 1, ne
     findex = 1.0_dtype

     if (ftype == 2) then
        ! Evaluate the filtered thickness of the given element
        ttmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))
           end if
        end do

        ! Compute the failure index
        findex = one + epsilon/ttmp - epsilon
     end if

     ! Evaluate the strain at the given point
     call evalShapeFunctions(xi, eta, ns, nxi, neta)
     call getElemGradient(i, n, ne, conn, X, nxi, neta, Xd)
     call getElemGradient(i, n, ne, conn, U, nxi, neta, Ud)

     ! Compute the inverse of Xd
     invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
     Jd(1,1) =  invdet*Xd(2,2)
     Jd(2,1) = -invdet*Xd(2,1)
     Jd(1,2) = -invdet*Xd(1,2)
     Jd(2,2) =  invdet*Xd(1,1)

     ! Evaluate the stress/strain
     call evalStrain(Jd, Ud, e)

     ! Evaluate the stress constraint from the given material type
     stress(i) = findex - (dot_product(e, h) + dot_product(e, matmul(G, e)))
  end do

end subroutine computeAllStress

subroutine computeLogStressSum( &
     ftype, xi, eta, n, ne, ntw, &
     conn, X, U, tconn, tweights, xdv, &
     epsilon, h, G, log_sum)
  ! Compute the sum of the log of the stress constraints in all the
  ! elements within the finite-element mesh.
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
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
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3), G(3,3)
  real(kind=dtype), intent(out) :: log_sum

  ! Temporary data used internally
  integer :: i, k
  real(kind=dtype) :: findex, ttmp, e(3), stress
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)
  real(kind=dtype) :: sum_pos, sum_neg

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Initialize the sums
  sum_pos = 0.0_dtype
  sum_neg = 0.0_dtype

  do i = 1, ne
     ! Compute the failure index
     findex = 1.0_dtype
     if (ftype == 2) then
        ! Evaluate the filtered thickness of the given element
        ttmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))
           end if
        end do

        ! Compute the failure index
        findex = one + epsilon/ttmp - epsilon
     end if

     ! Evaluate the strain at the given point
     call evalShapeFunctions(xi, eta, ns, nxi, neta)
     call getElemGradient(i, n, ne, conn, X, nxi, neta, Xd)
     call getElemGradient(i, n, ne, conn, U, nxi, neta, Ud)

     ! Compute the inverse of Xd
     invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
     Jd(1,1) =  invdet*Xd(2,2)
     Jd(2,1) = -invdet*Xd(2,1)
     Jd(1,2) = -invdet*Xd(1,2)
     Jd(2,2) =  invdet*Xd(1,1)

     ! Evaluate the stress/strain
     call evalStrain(Jd, Ud, e)

     ! Evaluate the stress constraint from the given material type
     stress = findex - dot_product(e, h) - dot_product(e, matmul(G, e))

     if (stress > 1.0) then
        sum_pos = sum_pos + log(stress)
     else
        sum_neg = sum_neg + log(stress)
     end if
  end do

  log_sum = sum_pos + sum_neg

end subroutine computeLogStressSum

subroutine computeMaxStep( &
     ftype, xi, eta, n, ne, ntw, &
     conn, X, U, Ustep, tconn, tweights, xdv, xstep, &
     epsilon, h, G, tau, alpha)
  ! Compute the sum of the log of the stress constraints in all the
  ! elements within the finite-element mesh.
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
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
  !
  ! Output:
  ! alpha:    the maximum step < 1.0 along the direction U + alpha*Ustep

  use precision
  implicit none

  integer, intent(in) :: ftype
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n), Ustep(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne), xstep(1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3), G(3,3), tau
  real(kind=dtype), intent(out) :: alpha

  ! Temporary data used internally
  integer :: i, k, l
  real(kind=dtype) :: findex, dfindex, ddfindex, e(3), es(3)
  real(kind=dtype) :: ttmp, dttmp, d2, d3
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)
  real(kind=dtype) :: a, b, c, discrim, r, r1, r2
  real(kind=dtype) :: p(ntw)

  ! Set the parameter
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Set the maximum step length
  alpha = 1.0_dtype

  do i = 1, ne
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

     ! Evaluate the shape functions and the strain
     call evalShapeFunctions(xi, eta, ns, nxi, neta)
     call getElemGradient(i, n, ne, conn, X, nxi, neta, Xd)

     ! Compute the inverse of Xd
     invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
     Jd(1,1) =  invdet*Xd(2,2)
     Jd(2,1) = -invdet*Xd(2,1)
     Jd(1,2) = -invdet*Xd(1,2)
     Jd(2,2) =  invdet*Xd(1,1)

     ! Evaluate the stress/strain
     call getElemGradient(i, n, ne, conn, U, nxi, neta, Ud)
     call evalStrain(Jd, Ud, e)

     ! Evaluate the stress/strain
     call getElemGradient(i, n, ne, conn, Ustep, nxi, neta, Ud)
     call evalStrain(Jd, Ud, es)

     ! Solve a quadratic equation to determine the max permissible
     ! step length
     a = 0.5*ddfindex - dot_product(es, matmul(G, es))
     b = dfindex - dot_product(es, h) - 2.0*dot_product(es, matmul(G, e))
     c = findex - dot_product(e, h) - dot_product(e, matmul(G, e))

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

end subroutine computeMaxStep

subroutine addLogStressSumDeriv( &
     ftype, xi, eta, n, ne, ntw, &
     conn, X, U, tconn, tweights, xdv, &
     epsilon, h, G, barrier, rx, ru)
  ! Add the derivative of the barrier term from the stress constraints
  ! to the vectors rx and ru
  !
  ! Input:
  ! ftype:    the type of failure parametrization to use
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
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

  integer, intent(in) :: ftype
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3), G(3,3), barrier
  real(kind=dtype), intent(inout) :: rx(1,ne), ru(2,n)

  ! Temporary data used internally
  integer :: i, k
  real(kind=dtype) :: findex, ttmp, fact, stress, e(3)
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), B(3,8)

  ! Set the value of unity!
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Set the thickness variable to zero
  ttmp = 0.0_dtype

  do i = 1, ne
     ! Compute the failure index
     findex = 1.0_dtype
     if (ftype == 2) then
        ! Evaluate the filtered thickness of the given element
        ttmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))
           end if
        end do

        ! Compute the failure index
        findex = one + epsilon/ttmp - epsilon
     end if

     ! Evaluate the shape functions and the strain
     call evalShapeFunctions(xi, eta, ns, nxi, neta)
     call getElemGradient(i, n, ne, conn, X, nxi, neta, Xd)
     call getElemGradient(i, n, ne, conn, U, nxi, neta, Ud)

     ! Compute the inverse of Xd
     invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
     Jd(1,1) =  invdet*Xd(2,2)
     Jd(2,1) = -invdet*Xd(2,1)
     Jd(1,2) = -invdet*Xd(1,2)
     Jd(2,2) =  invdet*Xd(1,1)

     ! Evaluate the stress/strain
     call evalStrain(Jd, Ud, e)

     ! Evaluate the derivative of the strain matrix
     call evalBmat(Jd, nxi, neta, B)

     ! Evaluate the stress constraint from the given material type
     stress = findex - dot_product(e, h) - dot_product(e, matmul(G, e))

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

     ! Add the terms to the derivative of the stress
     do k = 1, 4
        ru(1,conn(k,i)) = ru(1,conn(k,i)) + &
             fact*(dot_product(h, B(:,2*k-1)) &
             + 2.0*dot_product(e, matmul(G, B(:,2*k-1))))

        ru(2,conn(k,i)) = ru(2,conn(k,i)) + &
             fact*(dot_product(h, B(:,2*k)) &
             + 2.0*dot_product(e, matmul(G, B(:,2*k))))
     end do
  end do

end subroutine addLogStressSumDeriv

subroutine computeElemCmat( &
     index, ftype, xi, eta, n, ne, ntw, &
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
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3), G(3,3)
  real(kind=dtype), intent(inout) :: Cu(8,8)

  ! Temporary data used internally
  integer :: j, k
  real(kind=dtype) :: findex, ttmp, e(3), stress, fact
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

  ! Compute the stress factor
  stress = findex - dot_product(e, h) - dot_product(e, matmul(G, e))

  ! Compute the factor
  fact = one/stress**2

  ! Compute the failure-dependent material strain
  hi = h + 2.0*matmul(G, e)

  ! Compute the second derivatives w.r.t. u
  do j = 1, 8
     do k = 1, 8
        Cu(j,k) = Cu(j,k) + &
             fact*(dot_product(B(:,j), hi)*dot_product(B(:,k), hi) + &
             2.0*stress*dot_product(B(:,j), matmul(G, B(:,k))))
     end do
  end do

end subroutine computeElemCmat

subroutine computeCmat( &
     ftype, xi, eta, n, ne, ntw, &
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
  integer, intent(in) :: ftype
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3), G(3,3)
  integer, intent(in) :: nccols, crowp(n+1), ccols(nccols)
  real(kind=dtype), intent(inout) :: C(2,2,nccols)

  ! Temporary data used in the element computation
  integer :: i, ii, jj, jp
  real(kind=dtype) :: Cu(8,8)

  ! Zero all entries in the matrix
  C(:,:,:) = 0.0_dtype

  do i = 1, ne
     ! Evaluate the element stiffness matrix
     call computeElemCmat(i, ftype, xi, eta, n, ne, ntw, &
          conn, X, U, tconn, tweights, xdv, epsilon, h, G, Cu)

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
     ftype, xi, eta, n, ne, ntw, &
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
  integer, intent(in) :: ftype
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), xdv(1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3), G(3,3)
  integer, intent(in) :: ncols, rowp(ne+1), cols(ncols)
  real(kind=dtype), intent(inout) :: D(1,1,ncols)

  ! Temporary data used in the element computation
  integer :: i, k, ii, jj, ki, kj, jp
  real(kind=dtype) :: findex, ttmp, e(3)
  real(kind=dtype) :: stress, invs, tw
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4)

  ! Create a parameter for unity
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero all entries in the matrix
  D(:,:,:) = 0.0_dtype

  ! Zero out the filtered thickness
  ttmp = 0.0_dtype

  ! Loop over all the elements in the mesh
  do i = 1, ne
     ! Compute the failure index
     findex = 1.0_dtype
     if (ftype == 2) then
        ! Evaluate the filtered thickness of the given element
        ttmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))
           end if
        end do

        ! Compute the failure index
        findex = one + epsilon/ttmp - epsilon
     end if

     ! Evaluate the shape functions and the strain
     call evalShapeFunctions(xi, eta, ns, nxi, neta)
     call getElemGradient(i, n, ne, conn, X, nxi, neta, Xd)
     call getElemGradient(i, n, ne, conn, U, nxi, neta, Ud)

     ! Compute the inverse of Xd
     invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
     Jd(1,1) =  invdet*Xd(2,2)
     Jd(2,1) = -invdet*Xd(2,1)
     Jd(1,2) = -invdet*Xd(1,2)
     Jd(2,2) =  invdet*Xd(1,1)

     ! Evaluate the stress/strain
     call evalStrain(Jd, Ud, e)

     ! Evaluate the stress constraint from the given material type
     stress = findex - dot_product(e, h) - dot_product(e, matmul(G, e))

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

                          ! If ii == jj, add the additional second
                          ! derivative contribution
                          if (ii == jj) then
                             D(1,1,jp) = D(1,1,jp) + &
                                  2.0*epsilon*tweights(ii,i)/ &
                                  (stress*(ttmp**2)*(xdv(1,ki)**3))
                          end if
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

end subroutine computeDmat

subroutine addCmatOffDiagProduct( &
     ftype, xi, eta, n, ne, ntw, &
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
  integer, intent(in) :: ftype
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n), Ustep(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne)
  real(kind=dtype), intent(in) :: xdv(1,ne), xstep(1,ne)
  real(kind=dtype), intent(in) :: epsilon, h(3), G(3,3)
  real(kind=dtype), intent(in) :: barrier
  real(kind=dtype), intent(inout) :: rx(1,ne), ru(2,n)

  ! Temporary data used in the element computation
  integer :: i, k
  real(kind=dtype) :: findex, ttmp, stress
  real(kind=dtype) :: dttmp, invs, e(3), es(3)
  real(kind=dtype) :: Xd(2,2), Ud(2,2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), B(3,8)

  ! Create a parameter for unity
  real(kind=dtype), parameter :: one = 1.0_dtype

  ! Zero out the filtered thickness and material
  ttmp = 0.0_dtype
  dttmp = 0.0_dtype

  ! Loop over all the elements in the mesh
  do i = 1, ne
     ! Compute the failure index
     findex = 1.0_dtype
     if (ftype == 2) then
        ! Evaluate the filtered thickness of the given element
        ttmp = 0.0_dtype
        dttmp = 0.0_dtype
        do k = 1, ntw
           if (tconn(k,i) > 0) then
              ttmp = ttmp + tweights(k,i)/xdv(1,tconn(k,i))
              dttmp = dttmp - xstep(1,tconn(k,i))*tweights(k,i)/xdv(1,tconn(k,i))**2
           end if
        end do

        ! Compute the failure index
        findex = one + epsilon/ttmp - epsilon
     end if

     ! Evaluate the shape functions and the strain
     call evalShapeFunctions(xi, eta, ns, nxi, neta)
     call getElemGradient(i, n, ne, conn, X, nxi, neta, Xd)

     ! Compute the inverse of Xd
     invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
     Jd(1,1) =  invdet*Xd(2,2)
     Jd(2,1) = -invdet*Xd(2,1)
     Jd(1,2) = -invdet*Xd(1,2)
     Jd(2,2) =  invdet*Xd(1,1)

     ! Evaluate the stress/strain
     call getElemGradient(i, n, ne, conn, U, nxi, neta, Ud)
     call evalStrain(Jd, Ud, e)

     ! Compute the strain produced by the step
     call getElemGradient(i, n, ne, conn, Ustep, nxi, neta, Ud)
     call evalStrain(Jd, Ud, es)

     ! Evaluate the B-matrix
     call evalBmat(Jd, nxi, neta, B)

     ! Evaluate the stress constraint from the given material type
     stress = findex - dot_product(e, h) - dot_product(e, matmul(G, e))

     if (ftype == 2) then
        ! Add the contributions to the product with the state variables
        invs = - barrier*epsilon*(dot_product(es, h) + &
             2.0*dot_product(e, matmul(G, es)))/stress**2

        do k = 1, ntw
           if (tconn(k,i) > 0) then
              rx(1,tconn(k,i)) = rx(1,tconn(k,i)) + &
                   invs*tweights(k,i)/(ttmp*xdv(1,tconn(k,i)))**2
           end if
        end do

        invs = barrier*epsilon*dttmp/(ttmp*stress)**2

        do k = 1, 4
           ru(1,conn(k,i)) = ru(1,conn(k,i)) + &
                invs*(dot_product(h, B(:,2*k-1)) &
                + 2.0*dot_product(e, matmul(G, B(:,2*k-1))))

           ru(2,conn(k,i)) = ru(2,conn(k,i)) + &
                invs*(dot_product(h, B(:,2*k)) &
                + 2.0*dot_product(e, matmul(G, B(:,2*k))))
        end do
     end if
  end do

end subroutine addCmatOffDiagProduct
