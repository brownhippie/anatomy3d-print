# Figuro Android app

A thin native wrapper (Capacitor) around the webapp — its WebView loads
`https://web-production-d3d4b.up.railway.app` directly (see
`capacitor.config.ts`'s `server.url`), so every Railway deploy of the
webapp is what this app shows immediately. No separate app rebuild is
needed for ordinary content/feature changes — only for native-shell
changes (app name, icon, permissions, splash screen).

## Rebuilding the debug APK

```
cd android-app
npm install
npx cap sync android
cd android
./gradlew assembleDebug
```

The output lands at `android/app/build/outputs/apk/debug/app-debug.apk`.
Copy it to `../../webapp/static/downloads/figuro.apk` to update the
webapp's own download link.

This is a debug-signed build, not a Play Store release — it needs
"install from unknown sources" allowed to sideload. A real release
build needs its own signing keystore (not set up here) plus a Play
Console listing, both beyond what this repo currently has.

## App icon / splash screen

Still the generic Capacitor placeholder (`android/app/src/main/res/mipmap-*/ic_launcher*`,
`drawable*/splash.png`) — swap these for real artwork and re-run
`npx cap sync android` to pick them up, or use `@capacitor/assets` to
generate the full icon/splash set from one source image.
