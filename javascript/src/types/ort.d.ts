// "#ort" (package.json imports) is onnxruntime-web: its Node build in Node, its CPU-only build in browsers.
declare module "#ort" {
  export * from "onnxruntime-web";
}
