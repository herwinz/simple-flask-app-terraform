locals {
  tags                         = { azd-env-name : var.environment_name }
  sha                          = base64encode(sha256("${var.environment_name}${var.location}${data.azurerm_client_config.current.subscription_id}"))
  resource_token               = substr(replace(lower(local.sha), "[^A-Za-z0-9_]", ""), 0, 13)
}

# ------------------------------------------------------------------------------------------------------
# Deploy resource Group
# ------------------------------------------------------------------------------------------------------
resource "azurecaf_name" "rg_name" {
  name          = var.environment_name
  resource_type = "azurerm_resource_group"
  random_length = 0
  clean_input   = true
}

resource "azurerm_resource_group" "rg" {
  name     = azurecaf_name.rg_name.result
  location = var.location
  tags = local.tags
}

# ------------------------------------------------------------------------------------------------------
# Deploy log analytics
# ------------------------------------------------------------------------------------------------------
module "loganalytics" {
  source         = "./core/monitor/loganalytics"
  location       = var.location
  rg_name        = azurerm_resource_group.rg.name
  tags           = azurerm_resource_group.rg.tags
  resource_token = local.resource_token
}

# ------------------------------------------------------------------------------------------------------
# Deploy application insights
# ------------------------------------------------------------------------------------------------------
module "applicationinsights" {
  source           = "./core/monitor/applicationinsights"
  location         = var.location
  rg_name          = azurerm_resource_group.rg.name
  environment_name = var.environment_name
  workspace_id     = module.loganalytics.LOGANALYTICS_WORKSPACE_ID
  tags             = azurerm_resource_group.rg.tags
  resource_token   = local.resource_token
}

# ------------------------------------------------------------------------------------------------------
# Deploy app service plan
# ------------------------------------------------------------------------------------------------------
module "app_plan" {
  source = "./core/host/appserviceplan"
  rg_name = azurerm_resource_group.rg.name
  location = var.location
  resource_token = local.resource_token
  tags = azurerm_resource_group.rg.tags
  sku_name = "B1"
  os_type = "Linux"
}

# ------------------------------------------------------------------------------------------------------
# Deploy app service web app
# ------------------------------------------------------------------------------------------------------
module "web_app" {
  source = "./core/host/appservice/appservicepython"
  rg_name = azurerm_resource_group.rg.name
  location = var.location
  resource_token = local.resource_token
  tags = merge(local.tags, { azd-service-name : "python-app" })
  appservice_plan_id = module.app_plan.APPSERVICE_PLAN_ID
  app_command_line = ""
  app_settings = {}
  service_name = "web"
}
