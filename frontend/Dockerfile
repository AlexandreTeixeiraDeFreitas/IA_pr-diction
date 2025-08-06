FROM node:18-alpine
WORKDIR /app

# Copie uniquement package.json pour le cache
COPY package*.json ./
RUN npm install

EXPOSE 8080
