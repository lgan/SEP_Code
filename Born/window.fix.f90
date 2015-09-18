# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "<stdin>"

program window
  use sep
  implicit none

  logical :: verb
  logical :: abc
  integer :: nx,nz, ny
  integer :: iz,ix, iy
  real :: dx,dz, dy
  real :: oz,oy, ox, cut, minvel, val
  real, allocatable :: vel(:,:,:)

  call sep_init()

  call from_history("n1",nz); call from_history("o1",oz); call from_history("d1",dz)
  call from_history("n2",nx); call from_history("o2",ox); call from_history("d2",dx)
  call from_history("n3",ny); call from_history("o3",oy); call from_history("d3",dy)

  call from_param("cut",cut,1.)
  !----------------------------------------------------------------
  !! output files
  call to_history("n1",nz); call to_history("d1",dz); call to_history("o1",oz)
  call to_history("n2",nx); call to_history("d2",dx); call to_history("o2",ox)
  call to_history("n3",ny); call to_history("d3",dy); call to_history("o3",oy)


  !----------------------------------------------------------------
  call sep_close()
  !! expand domain

  allocate(vel(nz,nx,ny))
  call sep_read( vel )

  minvel=minval(vel)
write(0,*) "minvel",minvel
  do iy=1,ny
    do ix=1,nx
      do iz=1,nz

 val=vel(iz,ix,iy)

        if(val .lt. cut) then
    vel(iz,ix,iy)=0.0001
 else
    vel(iz,ix,iy)=1.
 end if

      end do
    end do
  end do




  call sep_write(vel)



  call exit(0)
end program window
