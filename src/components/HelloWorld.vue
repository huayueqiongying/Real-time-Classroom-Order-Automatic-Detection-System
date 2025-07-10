<template>
  <div class="hello">
    <h2>{{ msg }}</h2>
    <ul>
      <li v-for="(item, index) in items" :key="index">
        {{ item.text }}
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'HelloWorld',
  props: {
    msg: String
  },
  data() {
    return {
      items: [
        { text: '项目一' },
        { text: '项目二' },
        { text: '项目三' }
      ]
    }
  }
}
</script>

<style scoped>
.hello {
  margin: 20px auto;
  max-width: 400px;
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 5px;
}

h2 {
  color: #42b983;
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  padding: 8px;
  margin: 5px 0;
  background-color: #f5f5f5;
}
</style>
