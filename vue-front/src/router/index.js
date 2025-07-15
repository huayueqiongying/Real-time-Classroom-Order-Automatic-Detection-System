import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'
import About from '@/components/About'
import Login from '@/views/Login'
import Register from '@/views/Register'
import Dashboard from '@/views/Dashboard'
import CameraView from '@/views/CameraView'
import EventHandling from '@/views/EventHandling'
import Profile from '@/views/Profile'
import StudentList from '@/views/StudentList'
import DangerZone from '@/views/DangerZone'
//import AdminAuth from '@/views/AdminAuth'
//import AdminDashboard from '@/views/AdminDashboard'
//import WhiteList from '@/views/WhiteList'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      redirect: '/login'
    },
    {
      path: '/home',
      name: 'HelloWorld',
      component: HelloWorld
    },
    {
      path: '/about',
      name: 'About',
      component: About
    },
    {
      path: '/login',
      name: 'Login',
      component: Login
    },
    {
      path: '/register',
      name: 'Register',
      component: Register
    },
    // 教师界面路由
    {
      path: '/dashboard',
      name: 'Dashboard',
      component: Dashboard
    },
    {
      path: '/camera',
      name: 'CameraView',
      component: CameraView
    },
    {
      path: '/events',
      name: 'EventHandling',
      component: EventHandling
    },
    {
      path: '/profile',
      name: 'Profile',
      component: Profile
    },
    {
      path: '/students',
      name: 'StudentList',
      component: StudentList
    },
    // 管理员相关路由
    {
      path: '/danger-zone',
      name: 'DangerZone',
      component: DangerZone
    },
    //{
    //  path: '/admin',
    //  name: 'AdminDashboard',
    //  component: AdminDashboard
    //},
    //{
    //  path: '/whitelist',
    //  name: 'WhiteList',
    //  component: WhiteList
    //}
  ]
})
