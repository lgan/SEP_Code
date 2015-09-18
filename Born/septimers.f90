module timer_type_mod
 
type septimer
logical :: active
character(len=128) :: tag
real*8 :: tot_time
real*8 :: last_time
integer :: ncalls
end type
end module

module timers_mod
use sep_timer_type_mod
implicit none
type(septimer),private :: timers(100)
integer,private,save :: ilast

interface start_timer
  module procedure start_timer_num,start_timer_name,start_timer_struct
end interface

interface stop_timer
  module procedure stop_timer_num,stop_timer_name,stop_timer_struct
end interface

contains

subroutine init_timers()
  integer :: i
  do i=1,100
    call clear_time(timers(i))
  end do
  ilast=0
end  subroutine

subroutine print_timers()
  integer :: i
  do i=1,100
   call print_timer(timers(i))
  end do
end subroutine

subroutine print_timer(st)
  type(septimer) :: st
  if(st%active) then
    write(0,*) trim(st%tag), "=tag ncalls=",st%ncalls," tot_time=",st%tot_time
  end if
end subroutine

subroutine stop_timer_num(inum)
  integer :: inum
  call stop_timer(timers(inum))
end subroutine

subroutine stop_timer_struct(st)
  type(septimer) :: st
  st%tot_time=st%tot_time+get_time()-st%last_time
  st%ncalls=st%ncalls+1
end subroutine

subroutine start_timer_name(nm)
  character(len=*) :: nm
  integer ::i
  do i=1,100
    if(nm==timers(i)%tag) call start_timer(timers(i))
  end do
end subroutine

subroutine stop_timer_name(nm)
  character(len=*) :: nm
  integer ::i
  do i=1,100
    if(nm==timers(i)%tag) call stop_timer(timers(i))
  end do
end subroutine

subroutine start_timer_num(inum)
  integer :: inum
  call start_timer(timers(inum))
end subroutine

subroutine start_timer_struct(st)
  type(septimer)::st
  st%last_time=get_time()
end subroutine



logical function setup_next_timer(tag,inum)
integer,optional :: inum
character(len=*) :: tag
setup_next_timer=.false.
if(ilast >99) then
  write(0,*) "out of timers"
  return
end if
ilast=ilast+1
call setup_timer_num(ilast,tag)
setup_next_timer=.true.
if(present(inum)) inum=ilast
end  function

subroutine setup_timer_num(inum,tag)
integer :: inum
character(len=*) :: tag
call setup_timer(timers(inum),tag)
end subroutine


subroutine setup_timer(st,tag)
type(septimer) :: st
character(len=*) :: tag
st%tag=tag
st%tot_time=0.
st%active=.true.
st%ncalls=0
end subroutine

subroutine clear_time(st)
type(septimer) :: st
st%tag=""
st%active=.false.
st%tot_time=0.
end subroutine

real*8 function get_time()

CHARACTER (LEN=8) 	  :: date
CHARACTER (LEN=10) 	  :: time
CHARACTER (LEN=5) 	  :: zone
INTEGER, DIMENSION(8) 	  :: value         
!_____________________________________________________________
! value(1)=year
! value(2)=month
! value(3)=day
! value(4)=difference from UTC in minutes
! value(5)=hour
! value(6)=minute
! value(7)=second
! value(8)=millisecond
!_____________________________________________________________
! calculate the current time on the cpu clock in s.
! use Fortran90 intrinsic function 'date_and_time'
!_____________________________________________________________
CALL DATE_AND_TIME(date,time,zone,value)          

get_time=86400.*value(3)+3600.*value(5)+60.*value(6)+value(7)+.001*value(8)


END function

end module
