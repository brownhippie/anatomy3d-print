import type { CapacitorConfig } from '@capacitor/cli';

// server.url points the app's WebView directly at the live, deployed
// webapp instead of bundling a copy of its static files into the APK.
// This is deliberate: it means every Railway deploy of webapp/ is what
// this app shows immediately, with no separate app rebuild/resubmit
// needed for ordinary content/feature changes -- the "update" behavior
// requested. A new APK build is only needed for native-shell-level
// changes (app name/icon/permissions), not for the webapp itself.
const config: CapacitorConfig = {
  appId: 'com.anatomy3dprint.figuro',
  appName: 'Figuro',
  webDir: 'www',
  server: {
    url: 'https://web-production-d3d4b.up.railway.app',
    androidScheme: 'https',
  },
};

export default config;
