package example

import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.storage.InMemoryStatsStorage

object UiHost {
  def main(args: Array[String]): Unit = {
    //Initialize the user interface backend
    val uiServer = UIServer.getInstance()

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    val statsStorage = new InMemoryStatsStorage()         //Alternative: new FileStatsStorage(File), for saving and loading later

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage)

    uiServer.enableRemoteListener()
  }
}
