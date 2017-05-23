! The following code implements constraint and residual operations
! required by Kona Optimization Framework for reduced-space optimization.
!
! Copyright (c) Alp Dener 2015, All rights reserved.

subroutine computeStressConstraints( &
     xi, eta, n, ne, ntw, conn, X, U, &
     tconn, tweights, h, G, scons)
  ! Compute the stress constraints for each material in all of the
  ! elements in the finite-element mesh.
  !
  ! Input:
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  !
  ! Output:
  ! scons:   the values of the stress constraints

  use precision
  implicit none

  ! Input and output parameters
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), h(3), G(3,3)
  real(kind=dtype), intent(inout) :: scons(ne)

  ! Temporary data used internally
  integer :: i, k
  real(kind=dtype) :: one = 1.0_dtype
  real(kind=dtype) :: neg_one = -1.0_dtype
  real(kind=dtype) :: Xd(2,2), Ud(2, 2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), e(3), etmp(3)

  ! Evaluate the shape functions at the specified parametric point
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  ! Loop over elements
  do i = 1, ne
     ! Evaluate the filtered thickness and strain for this element
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evalaute the gradient of the position/displacements
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

     ! Evaluate the stress constraint from the given material type
     scons(i) = one - dot_product(e, h) - dot_product(e, matmul(G, e))
  end do

end subroutine computeStressConstraints

subroutine stressConstraintJacobianProduct( &
     xi, eta, n, ne, ntw, conn, X, U, &
     tconn, tweights, h, G, invec, outvec)
  ! Compute the linearized jacobian-vector product between the stress
  ! constraint jacobian and the given input vector, evaluated at the given
  ! displacements.
  !
  ! Input:
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! invec:    vector multiplying the jacobian
  !
  ! Output:
  ! outvec:   result of the jacobian-vector product

  use precision
  implicit none

  ! Input and output parameters
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), h(3), G(3,3)
  real(kind=dtype), intent(in) :: invec(2, n)
  real(kind=dtype), intent(inout) :: outvec(ne)

  ! Temporary data used internally
  integer :: i, j, k
  real(kind=dtype) :: neg_one = -1.0_dtype
  real(kind=dtype) :: two = 2.0_dtype
  real(kind=dtype) :: Xd(2,2), Ud(2, 2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), B(3,8), e(3), etmp(3)
  real(kind=dtype) :: invecLocal(8), row(8)

  ! Zero the output vec
  outvec(:) = 0.0_dtype

  ! Evaluate the shape functions
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  ! Loop over elements
  do i = 1, ne
     ! Evaluate the filtered strain required for the derivative of the
     ! failure constraint with respect to the state variables
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evalaute the gradient of the position/displacements
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

     ! Loop over parts of the filter and add the contribution from
     ! each element
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evaluate the element gradient of position/displacement
           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate strain derivatives w.r.t. displacement
           call evalBmat(Jd, nxi, neta, B)

           ! Extract the local displacements for the element and the
           ! local element block of the multiplying vector
           do j = 1, 4
              invecLocal(2*j-1) = invec(1,conn(j,tconn(k,i)))
              invecLocal(2*j) = invec(2,conn(j,tconn(k,i)))
           end do

           ! Calculate the non-zero values of the jacobian row
           row = neg_one*matmul(transpose(B), h) - two*matmul(transpose(B), matmul(G, e))

           ! Add the Jacobian-vector product contribution to the i-th
           ! element constraint
           outvec(i) = outvec(i) + tweights(k,i)*dot_product(row, invecLocal)
        end if
     end do
  end do

end subroutine stressConstraintJacobianProduct

subroutine stressConstraintJacobianTransProduct( &
     xi, eta, n, ne, ntw, conn, X, U, &
     tconn, tweights, h, G, invec, outvec)
  ! Compute the linearized transposed-jacobian-vector product between the
  ! stress constraint jacobian and the given input vector, evaluated at the
  ! given displacements.
  !
  ! Input:
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! invec:    vector multiplying the jacobian
  !
  ! Output:
  ! outvec:   result of the jacobian-vector product

  use precision
  implicit none

  ! Input and output parameters
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), h(3), G(3,3)
  real(kind=dtype), intent(in) :: invec(ne)
  real(kind=dtype), intent(inout) :: outvec(2, n)

  ! Temporary data used internally
  integer :: i, j, k
  real(kind=dtype) :: neg_one = -1.0_dtype
  real(kind=dtype) :: two = 2.0_dtype
  real(kind=dtype) :: Xd(2,2), Ud(2, 2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), B(3,8), e(3), etmp(3)
  real(kind=dtype) :: col(8)

  ! Zero the output vector
  outvec(:,:) = 0.0_dtype

  ! Evaluate the shape functions at the specified parametric point
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  ! Loop over elements
  do i = 1, ne
     ! Evaluate the filtered strain required for the derivative of the
     ! failure constraint with respect to the state variables
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evalaute the gradient of the position/displacements
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

     ! Loop over parts of the filter and add the contribution from
     ! each element
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evaluate the element gradient of position/displacement
           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate strain derivatives w.r.t. displacement
           call evalBmat(Jd, nxi, neta, B)

           ! Calculate the non-zero values of the jacobian row
           col = neg_one*matmul(transpose(B), h) - two*matmul(transpose(B), matmul(G, e))

           ! Calculate i-th element of the jacobian-vector product
           do j = 1, 4
              outvec(1,conn(j,tconn(k,i))) = &
                   outvec(1,conn(j,tconn(k,i))) + tweights(k,i)*invec(i)*col(2*j-1)
              outvec(2,conn(j,tconn(k,i))) = &
                   outvec(2,conn(j,tconn(k,i))) + tweights(k,i)*invec(i)*col(2*j)
           end do
        end if
     end do
  end do

end subroutine stressConstraintJacobianTransProduct

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

  vdot =  dot_product(x(1,:), y(1,:))

end subroutine vecDot

subroutine computeAggregate( &
  xi, eta, ncon, n, ne, ntw, conn, X, U, &
  tconn, tweights, h, G, rho, cons)
  ! Compute the stress constraints for each material in all of the
  ! elements in the finite-element mesh.
  !
  ! Input:
  ! xi, eta:  the xi/eta locations within all elements
  ! ncon:     the number of constraint aggregates
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! rho:      the aggregation penalty parameter
  !
  ! Output:
  ! cons:     the values of the stress constraint aggregates

  use precision
  implicit none

  ! Input and output parameters
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: ncon, n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), h(3), G(3,3)
  real(kind=dtype), intent(in) :: rho
  real(kind=dtype), intent(inout) :: cons(ncon)

  ! Data used internally in this function
  integer :: i, iks
  real(kind=dtype) :: kssum(ncon), ksmax(ncon)
  real(kind=dtype) :: stress(ne)

  ! Evaluate all of the constraints
  call computeStressConstraints(xi, eta, n, ne, ntw, conn, X, U, &
       tconn, tweights, h, G, stress)

  ! Zero the max and sums in the KS aggregate
  ksmax(:) = 0.0_dtype
  kssum(:) = 0.0_dtype

  if (ncon == 1) then
     ! Compute the maximum stress constraint values
     ksmax(1) = maxval(stress)

     ! Compute the sum of the exponential terms in the KS constraint
     ! aggregate
     do i = 1, ne
        kssum(1) = kssum(1) + exp(rho*(stress(i) - ksmax(1)))
     end do
  else
     ! Compute the maximum stress constraint values
     do iks = 1, ncon
        ksmax(iks) = maxval(stress(iks:ne:ncon))
     end do

     ! Compute the sum of the exponential terms for each of the KS
     ! constraint aggregates
     iks = 1
     do i = 1, ne
        ! Compute the contribution to the ks sum
        kssum(iks) = kssum(iks) + exp(rho*(stress(i) - ksmax(iks)))

        ! Increment iks until it exceeds the number of constraints
        iks = iks + 1
        if (iks > ncon) then
           iks = 1
        end if
     end do
  end if

  ! Finish computing the constrain aggregate
  cons(:) = ksmax + log(kssum)/rho

end subroutine computeAggregate

subroutine computeAggregateProduct( &
  xi, eta, ncon, n, ne, ntw, conn, X, U, &
  tconn, tweights, h, G, rho, invec, cons)
  ! Compute the stress constraints for each material in all of the
  ! elements in the finite-element mesh.
  !
  ! Input:
  ! xi, eta:  the xi/eta locations within all elements
  ! ncon:     the number of constraint aggregates
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! rho:      the aggregation penalty parameter
  ! invec:    the perturbation to the state vector
  !
  ! Output:
  ! cons:     results of the Jacobian-vector product

  use precision
  implicit none

  ! Input and output parameters
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: ncon, n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), h(3), G(3,3)
  real(kind=dtype), intent(in) :: rho, invec(2,n)
  real(kind=dtype), intent(inout) :: cons(ncon)

  ! Data used internally in this function
  integer :: i, iks, j, k
  real(kind=dtype) :: kssum(ncon), ksmax(ncon), ksweight
  real(kind=dtype) :: stress(ne)
  real(kind=dtype) :: neg_one = -1.0_dtype
  real(kind=dtype) :: two = 2.0_dtype
  real(kind=dtype) :: Xd(2,2), Ud(2, 2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), B(3,8), e(3), etmp(3)
  real(kind=dtype) :: invecLocal(8), row(8)

  ! Zero the output
  cons(:) = 0.0_dtype

  ! Evaluate all of the constraints
  call computeStressConstraints(xi, eta, n, ne, ntw, conn, X, U, &
       tconn, tweights, h, G, stress)

  ! Zero the max and sums in the KS aggregate
  ksmax(:) = 0.0_dtype
  kssum(:) = 0.0_dtype

  if (ncon == 1) then
     ! Compute the maximum stress constraint values
     ksmax(1) = maxval(stress)

     ! Compute the sum of the exponential terms in the KS constraint
     ! aggregate
     do i = 1, ne
        kssum(1) = kssum(1) + exp(rho*(stress(i) - ksmax(1)))
     end do
  else
     ! Compute the maximum stress constraint values
     do iks = 1, ncon
        ksmax(iks) = maxval(stress(iks:ne:ncon))
     end do

     ! Compute the sum of the exponential terms for each of the KS
     ! constraint aggregates
     iks = 1
     do i = 1, ne
        ! Compute the contribution to the ks sum
        kssum(iks) = kssum(iks) + exp(rho*(stress(i) - ksmax(iks)))

        ! Increment iks until it exceeds the number of constraints
        iks = iks + 1
        if (iks > ncon) then
           iks = 1
        end if
     end do
  end if

  ! Evaluate the shape functions
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  ! Loop over elements
  iks = 1
  do i = 1, ne
     ! Compute the ks weight on this gradient
     ksweight = exp(rho*(stress(i) - ksmax(iks)))/kssum(iks)

     ! Evaluate the filtered strain required for the derivative of the
     ! failure constraint with respect to the state variables
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evalaute the gradient of the position/displacements
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

     ! Loop over parts of the filter and add the contribution from
     ! each element
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evaluate the element gradient of position/displacement
           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate strain derivatives w.r.t. displacement
           call evalBmat(Jd, nxi, neta, B)

           ! Calculate the non-zero values of the jacobian row
           row = neg_one*matmul(transpose(B), h) - two*matmul(transpose(B), matmul(G, e))

           ! Extract the local displacements for the element and the
           ! local element block of the multiplying vector
           do j = 1, 4
              invecLocal(2*j-1) = invec(1,conn(j,tconn(k,i)))
              invecLocal(2*j) = invec(2,conn(j,tconn(k,i)))
           end do

           ! Add the Jacobian-vector product contribution to the i-th
           ! element constraint
           cons(iks) = cons(iks) + &
                ksweight*tweights(k,i)*dot_product(row, invecLocal)
        end if
     end do

     iks = iks + 1
     if (iks > ncon) then
        iks = 1
     end if
  end do

end subroutine computeAggregateProduct

subroutine computeAggregateTransProduct( &
  xi, eta, ncon, n, ne, ntw, conn, X, U, &
  tconn, tweights, h, G, rho, cons, outvec)
  ! Compute the stress constraints for each material in all of the
  ! elements in the finite-element mesh.
  !
  ! Input:
  ! xi, eta:  the xi/eta locations within all elements
  ! ncon:     the number of constraint aggregates
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  ! rho:      the aggregation penalty parameter
  ! cons:     the perturbation to the constraints
  !
  ! Output:
  ! outvec:   result of the transpose Jacobian-vector product

  use precision
  implicit none

  ! Input and output parameters
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: ncon, n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), h(3), G(3,3)
  real(kind=dtype), intent(in) :: rho, cons(ncon)
  real(kind=dtype), intent(inout) :: outvec(2,n)

  ! Data used internally in this function
  integer :: i, iks, j, k
  real(kind=dtype) :: kssum(ncon), ksmax(ncon), ksweight
  real(kind=dtype) :: stress(ne)
  real(kind=dtype) :: neg_one = -1.0_dtype
  real(kind=dtype) :: two = 2.0_dtype
  real(kind=dtype) :: Xd(2,2), Ud(2, 2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), B(3,8), e(3), etmp(3)
  real(kind=dtype) :: col(8)

  ! Zero the output
  outvec(:,:) = 0.0_dtype

  ! Evaluate all of the constraints
  call computeStressConstraints(xi, eta, n, ne, ntw, conn, X, U, &
       tconn, tweights, h, G, stress)

  ! Zero the max and sums in the KS aggregate
  ksmax(:) = 0.0_dtype
  kssum(:) = 0.0_dtype

  if (ncon == 1) then
     ! Compute the maximum stress constraint values
     ksmax(1) = maxval(stress)

     ! Compute the sum of the exponential terms in the KS constraint
     ! aggregate
     do i = 1, ne
        kssum(1) = kssum(1) + exp(rho*(stress(i) - ksmax(1)))
     end do
  else
     ! Compute the maximum stress constraint values
     do iks = 1, ncon
        ksmax(iks) = maxval(stress(iks:ne:ncon))
     end do

     ! Compute the sum of the exponential terms for each of the KS
     ! constraint aggregates
     iks = 1
     do i = 1, ne
        ! Compute the contribution to the ks sum
        kssum(iks) = kssum(iks) + exp(rho*(stress(i) - ksmax(iks)))

        ! Increment iks until it exceeds the number of constraints
        iks = iks + 1
        if (iks > ncon) then
           iks = 1
        end if
     end do
  end if

  ! Evaluate the shape functions
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  ! Loop over elements
  iks = 1
  do i = 1, ne
     ! Compute the ks weight on this gradient
     ksweight = exp(rho*(stress(i) - ksmax(iks)))/kssum(iks)

     ! Evaluate the filtered strain required for the derivative of the
     ! failure constraint with respect to the state variables
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evalaute the gradient of the position/displacements
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

     ! Loop over parts of the filter and add the contribution from
     ! each element
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evaluate the element gradient of position/displacement
           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate strain derivatives w.r.t. displacement
           call evalBmat(Jd, nxi, neta, B)

           ! Calculate the non-zero values of the jacobian row
           col = neg_one*matmul(transpose(B), h) - two*matmul(transpose(B), matmul(G, e))

           ! Add the Jacobian-vector product contribution to the i-th
           ! element constraint
           do j = 1, 4
              outvec(1,conn(j,tconn(k,i))) = &
                   outvec(1,conn(j,tconn(k,i))) + &
                   ksweight*tweights(k,i)*col(2*j-1)*cons(iks)

              outvec(2,conn(j,tconn(k,i))) = &
                   outvec(2,conn(j,tconn(k,i))) + &
                   ksweight*tweights(k,i)*col(2*j)*cons(iks)
           end do
        end if
     end do

     iks = iks + 1
     if (iks > ncon) then
        iks = 1
     end if
  end do

end subroutine computeAggregateTransProduct


subroutine computeStressConstraintJacobianAmat( &
     xi, eta, n, ne, ntw, conn, X, U, &
     tconn, tweights, h, G, A)
  ! Compute the stress constraint jacobian, evaluated at the given
  ! displacements, and store it in the given
  ! compressed sparse row data format.
  !
  ! Input:
  ! xi, eta:  the xi/eta locations within all elements
  ! n:        the number of nodes
  ! ne:       the number of elements
  ! ntw:      the maximum size of the thickness filter
  ! conn:     the connectivity of the underlying mesh
  ! X:        the nodal locations in the mesh
  ! U:        the nodal displacements
  ! tconn:    the thickness/material filter connectivity
  ! tweights: the thickness/material filter weights
  ! h:        the values of the linear terms
  ! G:        the values of the quadratic terms
  !
  ! Output:
  ! A:   the stress constraint Jacobian explicit matrix

  use precision
  implicit none

  ! Input and output parameters
  real(kind=dtype), intent(in) :: xi, eta
  integer, intent(in) :: n, ne, ntw, conn(4,ne)
  real(kind=dtype), intent(in) :: X(2,n), U(2,n)
  integer, intent(in) :: tconn(ntw,ne)
  real(kind=dtype), intent(in) :: tweights(ntw,ne), h(3), G(3,3)
  real(kind=dtype), intent(inout) :: A(ne, n, 2)

  ! Temporary data used internally
  integer :: i, j, k
  real(kind=dtype) :: zero = 0.0_dtype
  real(kind=dtype) :: neg_one = -1.0_dtype
  real(kind=dtype) :: two = 2.0_dtype
  real(kind=dtype) :: Xd(2,2), Ud(2, 2), Jd(2,2), invdet
  real(kind=dtype) :: ns(4), nxi(4), neta(4), B(3,8), e(3), etmp(3)
  real(kind=dtype) :: row(8)

  A(:,:,:) = zero
  ! Evaluate the shape functions
  call evalShapeFunctions(xi, eta, ns, nxi, neta)

  ! Loop over elements
  do i = 1, ne
     ! Evaluate the filtered strain required for the derivative of the
     ! failure constraint with respect to the state variables
     e(:) = 0.0_dtype
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evalaute the gradient of the position/displacements
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

     ! Loop over parts of the filter and add the contribution from
     ! each element
     do k = 1, ntw
        if (tconn(k,i) > 0) then
           ! Evaluate the element gradient of position/displacement
           call getElemGradient(tconn(k,i), n, ne, conn, X, nxi, neta, Xd)
           call getElemGradient(tconn(k,i), n, ne, conn, U, nxi, neta, Ud)

           ! Compute the inverse of Xd
           invdet = 1.0_dtype/(Xd(1,1)*Xd(2,2) - Xd(1,2)*Xd(2,1))
           Jd(1,1) =  invdet*Xd(2,2)
           Jd(2,1) = -invdet*Xd(2,1)
           Jd(1,2) = -invdet*Xd(1,2)
           Jd(2,2) =  invdet*Xd(1,1)

           ! Evaluate strain derivatives w.r.t. displacement
           call evalBmat(Jd, nxi, neta, B)

           ! Calculate the non-zero values of the jacobian row
           row = neg_one*matmul(transpose(B), h) - two*matmul(transpose(B), matmul(G, e))
           
           ! Add the row contribution to the i-th element constraint
           do j = 1, 4
              A(i, conn(j,tconn(k,i)), 1) = A(i, conn(j,tconn(k,i)), 1) + tweights(k,i)*row(2*j-1)
              A(i, conn(j,tconn(k,i)), 2) = A(i, conn(j,tconn(k,i)), 2) + tweights(k,i)*row(2*j)
           end do

        end if
     end do
  end do

end subroutine computeStressConstraintJacobianAmat

