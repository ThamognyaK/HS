/** @type {import('next').NextConfig} */
const nextConfig = {
  basePath: "/thailand-ris",
  assetPrefix: process.env.NODE_ENV === "production" ? "/thailand-ris" : "",
  reactStrictMode: true,
  swcMinify: true,
  optimizeFonts: false,
};

module.exports = nextConfig;
