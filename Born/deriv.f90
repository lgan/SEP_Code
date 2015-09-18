
program deriv
  use sep
  implicit none

  logical :: verb
  integer :: n1,n2,n3
  integer :: i1,i2,i3
  real    :: d1,d2,d3
  real    :: o1,o2,o3
  real, allocatable :: data(:,:,:),data_out(:,:,:)

  call sep_init()

  call from_history("n1",n1);  call from_history("o1",o1);  call from_history("d1",d1)
  call from_history("n2",n2);  call from_history("o2",o2);  call from_history("d2",d2)
  call from_history("n3",n3);  call from_history("o3",o3);  call from_history("d3",d3)

  call to_history("n1",n1);  call to_history("d1",d1);  call to_history("o1",o1)
  call to_history("n2",n2);  call to_history("d2",d2);  call to_history("o2",o2)
  call to_history("n3",n3);  call to_history("d3",d3);  call to_history("o3",o3)


  call sep_close()

  allocate(data(n1,n2,n3),data_out(n1,n2,n3))
  call sep_read( data )

  do i3=2,n3-1
    do i2=2,n2-1
      do i1=2,n1-1

	data_out(i1,i2,i3)=-6*data(i1,i2,i3)+data(i1+1,i2,i3)+data(i1-1,i2,i3)+data(i1,i2+1,i3)+data(i1,i2-1,i3)+data(i1,i2,i3+1)+data(i1,i2,i3-1)

      end do
    end do
  end do

  call sep_write(data_out)

!  call exit(0)

end program deriv








