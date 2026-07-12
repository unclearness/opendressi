package org.dressi.examples

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

class MainActivity : Activity(), NativeBridge.Listener {

    private lateinit var root: LinearLayout
    private lateinit var setupView: ScrollView
    private lateinit var runView: LinearLayout
    private lateinit var capsText: TextView
    private lateinit var exampleList: LinearLayout
    private lateinit var runTitle: TextView
    private lateinit var streamBar: LinearLayout
    private lateinit var logText: TextView
    private lateinit var logScroll: ScrollView
    private lateinit var stopButton: Button
    private lateinit var backButton: Button

    private var dataDir: File? = null
    private var caps: JSONObject? = null
    private val logLines = ArrayDeque<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        NativeBridge.setListener(this)

        thread {
            val dir = AssetInstaller.install(this)
            val capsJson = NativeBridge.queryDeviceCaps()
            runOnUiThread {
                dataDir = dir
                caps = runCatching { JSONObject(capsJson) }.getOrNull()
                capsText.text = formatCaps(capsJson)
                populateExamples()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        NativeBridge.setListener(null)
    }

    // ------------------------------- UI ---------------------------------

    private fun buildUi() {
        root = vbox()
        setupView = ScrollView(this)
        val setupBox = vbox()
        setupView.addView(setupBox)

        capsText = TextView(this).apply {
            text = "Querying Vulkan device..."
            typeface = Typeface.MONOSPACE
            textSize = 12f
            setPadding(24, 24, 24, 24)
        }
        exampleList = vbox()
        setupBox.addView(header("Device"))
        setupBox.addView(capsText)
        setupBox.addView(header("Examples"))
        setupBox.addView(exampleList)

        runView = vbox().apply { visibility = View.GONE }
        runTitle = TextView(this).apply {
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setPadding(24, 16, 24, 8)
        }
        val surface = SurfaceView(this)
        surface.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                NativeBridge.setSurface(holder.surface)
            }
            override fun surfaceChanged(holder: SurfaceHolder, format: Int,
                                        width: Int, height: Int) {
                NativeBridge.setSurface(holder.surface)
            }
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                NativeBridge.setSurface(null)
            }
        })
        streamBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        logText = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 10f
            setTextColor(Color.rgb(40, 40, 40))
            setPadding(16, 8, 16, 8)
        }
        logScroll = ScrollView(this)
        logScroll.addView(logText)

        stopButton = Button(this).apply {
            text = "Stop"
            setOnClickListener {
                NativeBridge.requestStop()
                isEnabled = false
                text = "Stopping..."
            }
        }
        backButton = Button(this).apply {
            text = "Back"
            isEnabled = false
            setOnClickListener { showSetup() }
        }
        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(stopButton, LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            addView(backButton, LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        }

        runView.addView(runTitle)
        runView.addView(surface, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 3f))
        runView.addView(HorizontalScrollView(this).apply {
            addView(streamBar)
        })
        runView.addView(logScroll, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 2f))
        runView.addView(buttonRow)

        root.addView(setupView, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.MATCH_PARENT))
        root.addView(runView, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.MATCH_PARENT))
        setContentView(root)
    }

    private fun vbox() = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
    }

    private fun header(text: String) = TextView(this).apply {
        this.text = text
        textSize = 18f
        setTypeface(null, Typeface.BOLD)
        setPadding(24, 32, 24, 8)
    }

    private fun formatCaps(json: String): String =
        runCatching {
            val o = JSONObject(json)
            if (o.has("error")) return "Vulkan error: ${o.getString("error")}"
            buildString {
                appendLine("${o.optString("deviceName")}  " +
                        "Vulkan ${o.optString("apiVersion")}")
                appendLine("geometryShader: " +
                        o.optBoolean("geometryShader"))
                appendLine("maxImageDimension2D: " +
                        o.optInt("maxImageDimension2D"))
                appendLine("maxColorAttachments: " +
                        o.optInt("maxColorAttachments"))
                appendLine("inputAttachments/stage: " +
                        o.optInt("maxPerStageDescriptorInputAttachments"))
                appendLine("sampledImages/stage: " +
                        o.optInt("maxPerStageDescriptorSampledImages"))
                append("storageImages/stage: " +
                        o.optInt("maxPerStageDescriptorStorageImages"))
            }
        }.getOrElse { "caps parse failed: $json" }

    private fun populateExamples() {
        exampleList.removeAllViews()
        val geometryShader = caps?.optBoolean("geometryShader", false) ?: false
        val data = dataDir ?: return
        for (entry in NativeBridge.listExamples()) {
            val name = entry.substringBefore('|')
            val needsGeom = entry.substringAfter('|') == "1"
            val missing = missingAsset(name, data)
            val button = Button(this).apply { text = name }
            when {
                needsGeom && !geometryShader -> {
                    button.isEnabled = false
                    button.text = "$name (geometryShader unavailable)"
                }
                missing != null -> {
                    button.isEnabled = false
                    button.text = "$name (missing asset: $missing)"
                }
                else -> button.setOnClickListener { startExample(name) }
            }
            exampleList.addView(button)
        }
    }

    private fun missingAsset(name: String, data: File): String? {
        fun need(path: String): String? =
            if (File(data, path).exists()) null else path
        return when (name) {
            "texture_optimization", "silhouette_optimization" ->
                need("bunny/bunny.obj")
            "pbr_shading", "pbr_material_optimization",
            "pbr_envmap_optimization" ->
                need("DamagedHelmet/glTF/DamagedHelmet.gltf")
                    ?: need("suburban_garden_512.exr")
            else -> null
        }
    }

    // ------------------------------ Runs --------------------------------

    // Native defaults == desktop defaults (verified on Adreno 740: no
    // device limit is hit; full optimization runs take ~5-8 min). Only the
    // paths differ from desktop. Tweak here for shorter runs.
    private fun defaultArgs(name: String): Array<String> {
        val data = dataDir!!
        val out = File(filesDir, "out/$name").absolutePath
        val mesh = File(data, "DamagedHelmet/glTF/DamagedHelmet.gltf")
        val env = File(data, "suburban_garden_512.exr")
        return when (name) {
            "image_fitting" -> arrayOf()
            "texture_optimization", "silhouette_optimization" -> arrayOf(
                "--data-dir=${File(data, "bunny")}", "--out-dir=$out")
            "pbr_shading", "pbr_material_optimization",
            "pbr_envmap_optimization" -> arrayOf(
                "--mesh=$mesh", "--env=$env", "--out-dir=$out")
            else -> arrayOf()
        }
    }

    private fun startExample(name: String) {
        val args = defaultArgs(name)
        logLines.clear()
        logText.text = ""
        if (!NativeBridge.startExample(name, args)) {
            Toast.makeText(this, "another example is still running",
                Toast.LENGTH_SHORT).show()
            return
        }
        runTitle.text = name
        stopButton.isEnabled = true
        stopButton.text = "Stop"
        backButton.isEnabled = false
        streamBar.removeAllViews()
        setupView.visibility = View.GONE
        runView.visibility = View.VISIBLE
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun showSetup() {
        runView.visibility = View.GONE
        setupView.visibility = View.VISIBLE
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    // --------------------- NativeBridge.Listener ------------------------

    override fun onLog(line: String) {
        runOnUiThread {
            logLines.addLast(line)
            while (logLines.size > 200) {
                logLines.removeFirst()
            }
            logText.text = logLines.joinToString("\n")
            logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    override fun onStreamsChanged(titles: Array<String>) {
        runOnUiThread {
            streamBar.removeAllViews()
            titles.forEachIndexed { i, title ->
                streamBar.addView(Button(this).apply {
                    text = title
                    textSize = 11f
                    setOnClickListener { NativeBridge.selectStream(i) }
                })
            }
        }
    }

    override fun onFinished(name: String, exitCode: Int) {
        runOnUiThread {
            val verdict = when (exitCode) {
                0 -> "OK"
                1 -> "quality gate not met (short run?)"
                else -> "error"
            }
            Toast.makeText(this, "$name finished: $verdict (exit $exitCode)",
                Toast.LENGTH_LONG).show()
            onLog("== $name finished with exit code $exitCode ==")
            stopButton.isEnabled = false
            backButton.isEnabled = true
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
}
