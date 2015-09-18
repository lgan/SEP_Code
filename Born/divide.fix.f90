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
  real :: oz,oy, ox, cut, minvel, val, ddy,ddx,ddz,d
  real, allocatable :: vel(:,:,:)

  call sep_init()

  call from_history("n1",nz); call from_history("o1",oz); call from_history("d1",dz)
  call from_history("n2",nx); call from_history("o2",ox); call from_history("d2",dx)
  call from_history("n3",ny); call from_history("o3",oy); call from_history("d3",dy)

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

  do iy=1,ny
    ddy=oy+iy*dy
    do ix=1,nx
      ddx=ox+ix*dx
      do iz=1,nz
      ddz=dz+iz*dz

        d=(ddy**2+ddx**2+ddz**2)


 vel(iz,ix,iy)=vel(iz,ix,iy)/d


      end do
    end do
  end do




  call sep_write(vel)



  call exit(0)
end program window
