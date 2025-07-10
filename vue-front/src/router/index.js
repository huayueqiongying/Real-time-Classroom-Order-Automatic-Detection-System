import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'
import About from '@/components/About'
import Login from '@/views/Login'  // 导入登录组件
import Register from '@/views/Register'  // 导入注册组件

Vue.use(Router)

export default new Router({
  routes: [
    // 将首页重定向到登录页
    {
      path: '/',
      redirect: '/login'
    },
    // 保留原有路由
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
    // 添加登录和注册路由
    {
      path: '/login',
      name: 'Login',
      component: Login
    },
    {
      path: '/register',
      name: 'Register',
      component: Register
    }
  ]
})
