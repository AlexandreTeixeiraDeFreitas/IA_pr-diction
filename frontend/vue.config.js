const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,

  devServer: {
    proxy: {
      '^/api': {
        target: 'http://backend:5000',
        changeOrigin: true,
        pathRewrite: { '^/api': '' }
      },
      '^/guest_token': {
        target: 'http://jwt_proxy:3001',
        changeOrigin: true
      }
    }
  }
})
