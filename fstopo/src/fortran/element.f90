! The following code implements a full space method for
! stress-constrained topology optimization. These subroutines should
! be called from python once they are wrapped with f2py.
!
! Copyright (c) Graeme J. Kennedy 2015, All rights reserved.

subroutine computePenalty(rho, qval, penalty)
  ! Given the density, compute the corresponding penalty

  use precision
  implicit none

  real(kind=dtype), intent(in) :: rho, qval
  real(kind=dtype), intent(out) :: penalty
  real(kind=dtype), parameter :: one = 1.0_dtype

  penalty = rho/(one + qval*(one - rho))

end subroutine computePenalty

subroutine computePenaltyDeriv(rho, qval, penalty, dpenalty)
  ! Given the density, compute the corresponding penalty and the
  ! derivative of the penalty with respect to rho

  use precision
  implicit none

  real(kind=dtype), intent(in) :: rho, qval
  real(kind=dtype), intent(out) :: penalty, dpenalty
  real(kind=dtype), parameter :: one = 1.0_dtype

  real(kind=dtype) :: tinv
  tinv = one/(one + qval*(one - rho))
  penalty = rho*tinv
  dpenalty = (qval + one)*tinv**2

end subroutine computePenaltyDeriv

subroutine computePenalty2ndDeriv(rho, qval, penalty, dpenalty, ddpenalty)
  ! Given the density, compute the corresponding penalty and the
  ! derivative of the penalty with respect to rho

  use precision
  implicit none

  real(kind=dtype), intent(in) :: rho, qval
  real(kind=dtype), intent(out) :: penalty, dpenalty, ddpenalty
  real(kind=dtype), parameter :: one = 1.0_dtype

  real(kind=dtype) :: tinv
  tinv = one/(one + qval*(one - rho))
  penalty = rho*tinv
  dpenalty = (qval + one)*tinv**2
  ddpenalty = 2.0*qval*(qval + one)*tinv**3

end subroutine computePenalty2ndDeriv

subroutine evalShapeFunctions(xi, eta, ns, nxi, neta)
  ! Evaluate bi-linear shape functions within the element
  !
  ! Input:
  ! xi, eta:   the parametric coordinate locations on [-1, 1]^2
  !
  ! Output:
  ! ns:    the shape functions
  ! nxi:   the derivative of the shape functions w.r.t. xi
  ! neta:  the derivative of the shape functions w.r.t. eta

  use precision
  implicit none

  real(kind=dtype), intent(in) :: xi, eta
  real(kind=dtype), intent(out) :: ns(4), nxi(4), neta(4)

  ! Evaluate the shape functions for the element
  ns(1) = 0.25*(1.0 - xi)*(1.0 - eta)
  ns(2) = 0.25*(1.0 + xi)*(1.0 - eta)
  ns(3) = 0.25*(1.0 - xi)*(1.0 + eta)
  ns(4) = 0.25*(1.0 + xi)*(1.0 + eta)

  ! Evaluate the derivative of the shape functions w.r.t. xi
  nxi(1) = 0.25*(eta - 1.0)
  nxi(2) = 0.25*(1.0 - eta)
  nxi(3) = -0.25*(1.0 + eta)
  nxi(4) = 0.25*(1.0 + eta)

  ! Evaluate the derivative of the shape functions w.r.t. eta
  neta(1) = 0.25*(xi - 1.0)
  neta(2) = -0.25*(1.0 + xi)
  neta(3) = 0.25*(1.0 - xi)
  neta(4) = 0.25*(1.0 + xi)

end subroutine evalShapeFunctions

subroutine getElemGradient(index, n, ne, conn, X, nxi, neta, Xd)
  ! Evaluate the derivative of X with respect to the local parametric
  ! coordinates.
  !
  ! Input:
  ! index:   the element index
  ! n:       the number of nodes
  ! ne:      the number of elements
  ! conn:    the element connectivity
  ! X:       the nodal locations
  ! nxi:     the derivative of the shape functions w.r.t. xi
  ! neta:    the derivative of the shape functions w.r.t. eta
  ! Xd:      the gradient w.r.t. the local coordinate system

  use precision
  implicit none

  ! The input/output declarations
  integer, intent(in) :: index, n, ne, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n)
  real(kind=dtype), intent(in) :: nxi(4), neta(4)
  real(kind=dtype), intent(out) :: Xd(2,2)

  ! Index counter
  integer :: k

  do k = 1, 2
     Xd(k,1) = ( &
          nxi(1)*X(k, conn(1, index)) + &
          nxi(2)*X(k, conn(2, index)) + &
          nxi(3)*X(k, conn(3, index)) + &
          nxi(4)*X(k, conn(4, index)))

     Xd(k,2) = ( &
          neta(1)*X(k, conn(1, index)) + &
          neta(2)*X(k, conn(2, index)) + &
          neta(3)*X(k, conn(3, index)) + &
          neta(4)*X(k, conn(4, index)))
  end do

end subroutine getElemGradient

subroutine evalStrain(Jd, Ud, e)
  ! Given the displacement gradient ud, evaluate the strain.
  ! This uses the chain rule in the following manner:
  !
  ! U,d = U,x*X,d  ==> U,x = U,d*{X,d}^{-1} = U,d*J
  !
  ! Input:
  ! J:    the inverse of the derivative of the coords w.r.t. xi, eta
  ! Ud:   the derivative of the u,v displacements w.r.t. xi, eta
  !
  ! Output:
  ! e:    the strain

  use precision
  implicit none

  ! Input/output declarations
  real(kind=dtype), intent(in) :: Jd(2,2), Ud(2,2)
  real(kind=dtype), intent(out) :: e(3)

  ! The derivatives of the displacements
  real(kind=dtype) :: ux, uy, vx, vy

  ux = Ud(1,1)*Jd(1,1) + Ud(1,2)*Jd(2,1)
  uy = Ud(1,1)*Jd(1,2) + Ud(1,2)*Jd(2,2)

  vx = Ud(2,1)*Jd(1,1) + Ud(2,2)*Jd(2,1)
  vy = Ud(2,1)*Jd(1,2) + Ud(2,2)*Jd(2,2)

  e(1) = ux
  e(2) = vy
  e(3) = uy + vx

end subroutine evalStrain

subroutine evalBmat(Jd, nxi, neta, B)
  ! Given the matrix J = {Xd}^{-1}, and the derivatives of the shape
  ! functions, compute the derivative of the strain with respect to
  ! the displacements.
  !
  ! Input:
  ! J:    the inverse of the
  ! nxi:  the derivative of the shape functions w.r.t. xi
  ! neta: the derivative of the shape functions w.r.t. eta
  !
  ! Output:
  ! B:    the derivative of the strain with respect to the displacements

  use precision
  implicit none

  ! In/out declarations
  real(kind=dtype), intent(in) :: Jd(2,2), nxi(4), neta(4)
  real(kind=dtype), intent(out) :: B(3,8)

  ! Temporary values
  integer :: i
  real(kind=dtype) :: dx, dy

  ! Zero the values
  B(:,:) = 0.0_dtype

  do i = 1,4
     dx = nxi(i)*Jd(1,1) + neta(i)*Jd(2,1)
     dy = nxi(i)*Jd(1,2) + neta(i)*Jd(2,2)

     ! Compute the derivative w.r.t. u
     B(1,2*i-1) = dx
     B(3,2*i-1) = dy

     ! Add the derivative w.r.t. v
     B(2,2*i) = dy
     B(3,2*i) = dx
  end do

end subroutine evalBmat

subroutine computeElemKmat(index, n, ne, conn, X, C, Ke)
  ! Evaluate the stiffness matrix for the given element number with
  ! the specified modulus of elasticity.
  !
  ! Input:
  ! index:  the element index in the connectivity array
  ! n:      the number of nodes
  ! ne:     the number of elements
  ! conn:   the connectivity
  ! X:      the x/y node locations
  ! E:      the Young's modulus
  ! nu:     the Poisson ratio
  !
  ! Output:
  ! Ke:     the element stiffness matrix

  use precision
  implicit none

  integer, intent(in) :: index, n, ne, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), C(3,3)
  real(kind=dtype), intent(inout) :: Ke(8,8)

  ! Temporary data used in the element calculation
  integer :: i, j, ii, jj
  real(kind=dtype) :: B(3,8), s(3)
  real(kind=dtype) :: Xd(2,2), Jd(2,2), ns(4), nxi(4), neta(4)
  real(kind=dtype) :: quadpts(2), quadwts(2)
  real(kind=dtype) :: det, invdet, h

  ! Set the Gauss quadrature point/weight values
  quadpts(1) = -0.577350269189626_dtype
  quadpts(2) = 0.577350269189626_dtype
  quadwts(1) = 1.0_dtype
  quadwts(2) = 1.0_dtype

  ! Zero all the elements in the stiffness matrix
  Ke(:,:) = 0.0_dtype

  do j = 1,2
     do i = 1,2
        ! Evaluate the shape functions
        call evalShapeFunctions(quadpts(i), quadpts(j), ns, nxi, neta)

        ! Evaluate the Jacobian of the residuals
        call getElemGradient(index, n, ne, conn, X, nxi, neta, Xd)

        ! Compute J = Xd^{-1}
        det = Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1)
        invdet = 1.0_dtype/det
        Jd(1,1) =  invdet*Xd(2,2)
        Jd(2,1) = -invdet*Xd(2,1)
        Jd(1,2) = -invdet*Xd(1,2)
        Jd(2,2) =  invdet*Xd(1,1)

        ! Compute the quadrature weight at this point
        h = quadwts(i)*quadwts(j)*det

        ! Evaluate the derivative of the strain matrix
        call evalBmat(Jd, nxi, neta, B)

        do jj = 1,8
           s(1) = C(1,1)*B(1,jj) + C(1,2)*B(2,jj) + C(1,3)*B(3,jj)
           s(2) = C(2,1)*B(1,jj) + C(2,2)*B(2,jj) + C(2,3)*B(3,jj)
           s(3) = C(3,1)*B(1,jj) + C(3,2)*B(2,jj) + C(3,3)*B(3,jj)

           do ii = 1,8
              Ke(ii, jj) = Ke(ii, jj) + &
                   h*(s(1)*B(1,ii) + s(2)*B(2,ii) + s(3)*B(3,ii))
           end do
        end do
     end do
  end do

end subroutine computeElemKmat
