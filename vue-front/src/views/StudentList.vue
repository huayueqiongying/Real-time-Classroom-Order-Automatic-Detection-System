<template>
  <div class="student-list">
    <h2>查看人员名单</h2>
    <div class="search-container">
      <input
        type="text"
        v-model="searchQuery"
        placeholder="输入姓名或ID搜索"
        class="search-input"
        @keyup.enter="handleSearch"
      >
      <button @click="handleSearch" class="search-button">查询</button>
    </div>

    <div class="user-list-container">
      <table class="user-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>姓名</th>
            <th>性别</th>
            <th>电话</th>
            <th>角色</th>
            <th>注册时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="user in filteredUsers" :key="user.id">
            <td>{{ user.user_id }}</td>
            <td>{{ user.name }}</td>
            <td>{{ user.gender === 'male' ? '男' : '女' }}</td>
            <td>{{ user.phone }}</td>
            <td>{{ user.role === 'student' ? '学生' : '教师' }}</td>
            <td>{{ formatDate(user.created_at) }}</td>
            <td>
              <button @click="confirmDelete(user)" class="delete-button">删除</button>
            </td>
          </tr>
        </tbody>
      </table>

      <div v-if="loading" class="loading">加载中...</div>
      <div v-if="error" class="error">{{ error }}</div>
      <div v-if="!loading && filteredUsers.length === 0" class="no-data">
        没有找到匹配的用户
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'StudentList',
  data() {
    return {
      searchQuery: '',
      users: [],
      loading: false,
      error: null
    }
  },
  computed: {
    filteredUsers() {
      if (!this.searchQuery) {
        return this.users;
      }
      const query = this.searchQuery.toLowerCase();
      return this.users.filter(user =>
        user.name.toLowerCase().includes(query) ||
        user.user_id.toLowerCase().includes(query)
      );
    }
  },
  methods: {
    async fetchUsers() {
      this.loading = true;
      this.error = null;
      try {
        const response = await fetch('http://localhost:3000/api/user-data/users');
        if (!response.ok) {
          throw new Error('获取用户数据失败');
        }
        const data = await response.json();
        this.users = Array.isArray(data) ? data : [];
      } catch (err) {
        console.error('获取用户数据出错:', err);
        this.error = err.message;
      } finally {
        this.loading = false;
      }
    },
    handleSearch() {
      // 搜索逻辑已经在计算属性filteredUsers中实现
    },
    formatDate(timestamp) {
      if (!timestamp) return '';
      const date = new Date(timestamp);
      return date.toLocaleString();
    },

    async confirmDelete(user) {
      if (confirm(`确定要删除 ${user.name} (${user.user_id}) 吗？`)) {
        await this.deleteUser(user.id);
      }
    },
    async deleteUser(userId) {
      try {
        const response = await fetch(`http://localhost:3000/api/user-data/users/${userId}`, {
          method: 'DELETE'
        });

        if (!response.ok) {
          throw new Error('删除失败');
        }

        // 删除成功后刷新列表
        this.fetchUsers();
      } catch (err) {
        console.error('删除用户出错:', err);
        this.error = err.message;
      }
    }
  },
  mounted() {
    this.fetchUsers();
  }
}
</script>

<style scoped>
.student-list {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.search-container {
  display: flex;
  margin-bottom: 20px;
  max-width: 500px;
}

.search-input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px 0 0 4px;
  font-size: 14px;
  outline: none;
}

.search-button {
  padding: 8px 16px;
  background-color: #409eff;
  color: white;
  border: none;
  border-radius: 0 4px 4px 0;
  cursor: pointer;
  font-size: 14px;
}

/* 删除按钮样式 */
.delete-button {
  padding: 6px 12px;
  background-color: #f56c6c;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.delete-button:hover {
  background-color: #e74c3c;
}

.search-button:hover {
  background-color: #66b1ff;
}

.user-list-container {
  margin-top: 20px;
  overflow-x: auto;
}

.user-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}

.user-table th, .user-table td {
  border: 1px solid #ddd;
  padding: 12px;
  text-align: left;
}

.user-table th {
  background-color: #f2f2f2;
  font-weight: bold;
}

.user-table tr:nth-child(even) {
  background-color: #f9f9f9;
}

.user-table tr:hover {
  background-color: #f1f1f1;
}

.loading, .error, .no-data {
  padding: 20px;
  text-align: center;
  color: #666;
}

.error {
  color: #f56c6c;
}

.user-image {
  width: 50px;
  height: 50px;
  object-fit: cover;
  border-radius: 50%;
}
</style>
